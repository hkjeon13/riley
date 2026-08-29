#!/usr/bin/env python3
"""Fail-closed C02-P1 checker for a binary-bound reconstructed prior baseline.

This checker establishes a narrowly scoped fact: two independently captured
builds from one annotated Git tag agree on their recorded server binary,
bundle, and OCI leaf artifacts.  It does *not* attest that an older binary was
distributed or that an historical stable artifact exists.  It requires the P1
vocabulary exactly:

* ``baseline_kind = reconstructed-tag-baseline``
* ``provenance_class = reconstructed-from-source``
* ``historical_distribution = not-attested``
* ``historical_stable_artifact_status = unavailable``
* ``was_previously_shipped = false``

All raw leaves are ``provenance_v2_common.EvidenceDescriptor`` values with a
path, SHA-256, and byte length.  The common module is mandatory: it supplies
the no-follow, pinned-directory-FD reader used for every evidence read.

``oci.archive`` is deliberately treated as a fully checksummed opaque leaf.
This checker does not inspect Docker-save members or prove that the archive's
Config member digest equals the raw Docker ``Id``.  That content-level claim
requires a separately reviewed streaming archive consumer; it is not implied
by the canonical image-inspect cross-binding below.

Likewise, the source archive is a checksummed opaque leaf bound to raw Git tag
object/target observations, not proof that its contents were generated from
the target commit.  Replaying the source-archive generation belongs to a
separate builder-stage attestation.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence

try:
    import provenance_v2_common as common
except ImportError as error:  # pragma: no cover - integration-only guard
    raise SystemExit(
        "check_reconstructed_prior_baseline.py requires reviewed "
        "ci/release/provenance_v2_common.py"
    ) from error


MANIFEST_VERSION = "riley.reconstructed-prior-baseline.v2"
LEGACY_MANIFEST_VERSION = "riley.reconstructed-prior-baseline.v1"
BUILD_RECEIPT_VERSION = "riley.reconstructed-prior-build.v2"
GIT_TAG_OBJECT_VERSION = "riley.git-tag-object-observation.v1"
GIT_TAG_TARGET_VERSION = "riley.git-tag-target-observation.v1"
RECIPE_INSPECT_VERSION = "riley.reconstructed-build-recipe-inspect.v2"
IMAGE_INSPECT_VERSION = "riley.reconstructed-oci-image-inspect.v1"
CHECK_REPORT_VERSION = "riley.reconstructed-prior-baseline-check.v2"

BASELINE_KIND = "reconstructed-tag-baseline"
PROVENANCE_CLASS = "reconstructed-from-source"
HISTORICAL_DISTRIBUTION = "not-attested"
HISTORICAL_STABLE_ARTIFACT_STATUS = "unavailable"
WAS_PREVIOUSLY_SHIPPED = False

TAG_NAME_RE = re.compile(
    r"riley-(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)-rc[1-9][0-9]*"
)
IMAGE_ID_RE = re.compile(r"sha256:([0-9a-f]{64})")

CHECK_NAMES = (
    "reconstructed-tag-baseline-only",
    "provenance-class-reconstructed-from-source",
    "historical-distribution-not-attested",
    "historical-stable-artifact-unavailable",
    "not-previously-shipped",
    "git-tag-object-target-binding",
    "source-archive-binding",
    "independent-a-b-reconstruction",
    "recipe-inspect-binding",
    "runtime-image-inspect-raw-binding",
    "oci-image-inspect-binding",
    "binary-a-b-equality",
    "bundle-a-b-equality",
    "oci-archive-a-b-equality",
    "oci-layout-a-b-equality",
    "oci-manifest-a-b-equality",
    "oci-image-a-b-equality",
)


class BaselineError(common.ProvenanceV2Error):
    """A baseline cannot establish the C02-P1 rollback prerequisite."""


def _fail(code: str, message: str) -> NoReturn:
    error = BaselineError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


Descriptor = common.EvidenceDescriptor


@dataclass(frozen=True)
class SourceBinding:
    tag_name: str
    tag_object: Descriptor
    tag_target: Descriptor
    archive: Descriptor

    def as_json(self) -> dict[str, Any]:
        return {
            "tag_name": self.tag_name,
            "tag_object": self.tag_object.as_json(),
            "tag_target": self.tag_target.as_json(),
            "archive": self.archive.as_json(),
        }


@dataclass(frozen=True)
class TagIdentity:
    tag_name: str
    tag_ref: str
    tag_object_sha1: str
    target_commit_sha1: str


@dataclass(frozen=True)
class OciArtifacts:
    archive: Descriptor
    layout: Descriptor
    manifest: Descriptor

    def as_json(self) -> dict[str, Any]:
        return {
            "archive": self.archive.as_json(),
            "layout": self.layout.as_json(),
            "manifest": self.manifest.as_json(),
        }


@dataclass(frozen=True)
class ArtifactSet:
    binary: Descriptor
    bundle: Descriptor
    oci: OciArtifacts

    def as_json(self) -> dict[str, Any]:
        return {
            "binary": self.binary.as_json(),
            "bundle": self.bundle.as_json(),
            "oci": self.oci.as_json(),
        }


@dataclass(frozen=True)
class BuildReceipt:
    baseline_id: str
    reconstruction_id: str
    source: SourceBinding
    recipe_inspect: Descriptor
    image_inspect: Descriptor
    runtime_image_inspect_raw: Descriptor
    artifacts: ArtifactSet


@dataclass(frozen=True)
class RecipeInspect:
    baseline_id: str
    reconstruction_id: str
    source: SourceBinding
    recipe: Descriptor
    binary: Descriptor
    bundle: Descriptor


@dataclass(frozen=True)
class ImageInspect:
    baseline_id: str
    reconstruction_id: str
    source: SourceBinding
    runtime_image_inspect_raw: Descriptor
    archive: Descriptor
    layout: Descriptor
    manifest: Descriptor
    image_id: str


@dataclass(frozen=True)
class EqualityDescriptor:
    a: Descriptor
    b: Descriptor
    sha256: str

    def as_json(self) -> dict[str, Any]:
        return {"a": self.a.as_json(), "b": self.b.as_json(), "sha256": self.sha256}


@dataclass(frozen=True)
class Equality:
    binary: EqualityDescriptor
    bundle: EqualityDescriptor
    oci_archive: EqualityDescriptor
    oci_layout: EqualityDescriptor
    oci_manifest: EqualityDescriptor
    oci_image_id: str

    def as_json(self) -> dict[str, Any]:
        return {
            "binary": self.binary.as_json(),
            "bundle": self.bundle.as_json(),
            "oci_archive": self.oci_archive.as_json(),
            "oci_layout": self.oci_layout.as_json(),
            "oci_manifest": self.oci_manifest.as_json(),
            "oci_image": {
                "a": self.oci_image_id,
                "b": self.oci_image_id,
                "image_id": self.oci_image_id,
            },
        }


@dataclass(frozen=True)
class BaselineManifest:
    baseline_id: str
    source: SourceBinding
    a_receipt: Descriptor
    b_receipt: Descriptor
    equality: Equality


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = sorted(value) if isinstance(value, dict) else []
        _fail(
            "unknown-or-missing-field",
            f"{label} fields differ; expected={sorted(fields)}, actual={actual}",
        )
    return value


def _string(value: Any, label: str, *, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > maximum:
        _fail("invalid-string", f"{label} must be a bounded non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or common.SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail("invalid-sha256", f"{label} must be a non-zero lowercase SHA-256")
    return value


def _git_sha1(value: Any, label: str) -> str:
    value = _string(value, label, maximum=40)
    if re.fullmatch(r"[0-9a-f]{40}", value) is None or value == "0" * 40:
        _fail("invalid-git-sha1", f"{label} must be a non-zero lowercase Git SHA-1")
    return value


def _descriptor(value: Any, label: str) -> Descriptor:
    descriptor = common.parse_descriptor(value, label)
    if descriptor.byte_length < 1:
        _fail("empty-evidence-descriptor", f"{label}.byte_length must be positive")
    return descriptor


def _tag_name(value: Any, label: str) -> str:
    value = _string(value, label, maximum=128)
    if TAG_NAME_RE.fullmatch(value) is None:
        _fail("non-tag-baseline", f"{label} must be a normalized Riley RC tag")
    return value


def _source(value: Any, label: str) -> SourceBinding:
    row = _exact(value, {"tag_name", "tag_object", "tag_target", "archive"}, label)
    return SourceBinding(
        tag_name=_tag_name(row["tag_name"], f"{label}.tag_name"),
        tag_object=_descriptor(row["tag_object"], f"{label}.tag_object"),
        tag_target=_descriptor(row["tag_target"], f"{label}.tag_target"),
        archive=_descriptor(row["archive"], f"{label}.archive"),
    )


def _oci(value: Any, label: str) -> OciArtifacts:
    row = _exact(value, {"archive", "layout", "manifest"}, label)
    archive = _descriptor(row["archive"], f"{label}.archive")
    layout = _descriptor(row["layout"], f"{label}.layout")
    manifest = _descriptor(row["manifest"], f"{label}.manifest")
    if len({archive.path, layout.path, manifest.path}) != 3:
        _fail("duplicate-evidence-path", f"{label} archive, layout, and manifest must be distinct")
    return OciArtifacts(archive=archive, layout=layout, manifest=manifest)


def _artifacts(value: Any, label: str) -> ArtifactSet:
    row = _exact(value, {"binary", "bundle", "oci"}, label)
    binary = _descriptor(row["binary"], f"{label}.binary")
    bundle = _descriptor(row["bundle"], f"{label}.bundle")
    oci = _oci(row["oci"], f"{label}.oci")
    if len({binary.path, bundle.path, oci.archive.path, oci.layout.path, oci.manifest.path}) != 5:
        _fail(
            "duplicate-evidence-path",
            f"{label} binary, bundle, and OCI leaves must be distinct",
        )
    return ArtifactSet(binary=binary, bundle=bundle, oci=oci)


def _build_receipt(value: Any, label: str) -> BuildReceipt:
    row = _exact(
        value,
        {
            "schema_version",
            "baseline_id",
            "reconstruction_id",
            "source",
            "recipe_inspect",
            "image_inspect",
            "runtime_image_inspect_raw",
            "artifacts",
        },
        label,
    )
    if row["schema_version"] != BUILD_RECEIPT_VERSION:
        _fail("unsupported-build-receipt-version", f"{label}.schema_version is unsupported")
    reconstruction_id = row["reconstruction_id"]
    if reconstruction_id not in {"a", "b"}:
        _fail("invalid-reconstruction-id", f"{label}.reconstruction_id must be a or b")
    return BuildReceipt(
        baseline_id=_string(row["baseline_id"], f"{label}.baseline_id", maximum=160),
        reconstruction_id=reconstruction_id,
        source=_source(row["source"], f"{label}.source"),
        recipe_inspect=_descriptor(row["recipe_inspect"], f"{label}.recipe_inspect"),
        image_inspect=_descriptor(row["image_inspect"], f"{label}.image_inspect"),
        runtime_image_inspect_raw=_descriptor(
            row["runtime_image_inspect_raw"], f"{label}.runtime_image_inspect_raw"
        ),
        artifacts=_artifacts(row["artifacts"], f"{label}.artifacts"),
    )


def _recipe_inspect(value: Any, label: str) -> RecipeInspect:
    row = _exact(
        value,
        {
            "schema_version",
            "baseline_id",
            "reconstruction_id",
            "source",
            "recipe",
            "binary",
            "bundle",
        },
        label,
    )
    if row["schema_version"] != RECIPE_INSPECT_VERSION:
        _fail("unsupported-recipe-inspect-version", f"{label}.schema_version is unsupported")
    reconstruction_id = row["reconstruction_id"]
    if reconstruction_id not in {"a", "b"}:
        _fail("invalid-reconstruction-id", f"{label}.reconstruction_id must be a or b")
    return RecipeInspect(
        baseline_id=_string(row["baseline_id"], f"{label}.baseline_id", maximum=160),
        reconstruction_id=reconstruction_id,
        source=_source(row["source"], f"{label}.source"),
        recipe=_descriptor(row["recipe"], f"{label}.recipe"),
        binary=_descriptor(row["binary"], f"{label}.binary"),
        bundle=_descriptor(row["bundle"], f"{label}.bundle"),
    )


def _image_id(value: Any, label: str) -> str:
    value = _string(value, label, maximum=71)
    match = IMAGE_ID_RE.fullmatch(value)
    if match is None:
        _fail("invalid-oci-image-id", f"{label} must be a lowercase OCI SHA-256 image ID")
    _sha256(match.group(1), label)
    return value


def _image_inspect(value: Any, label: str) -> ImageInspect:
    row = _exact(
        value,
        {
            "schema_version",
            "baseline_id",
            "reconstruction_id",
            "source",
            "runtime_image_inspect_raw",
            "oci_archive",
            "oci_layout",
            "oci_manifest",
            "image_id",
        },
        label,
    )
    if row["schema_version"] != IMAGE_INSPECT_VERSION:
        _fail("unsupported-image-inspect-version", f"{label}.schema_version is unsupported")
    reconstruction_id = row["reconstruction_id"]
    if reconstruction_id not in {"a", "b"}:
        _fail("invalid-reconstruction-id", f"{label}.reconstruction_id must be a or b")
    return ImageInspect(
        baseline_id=_string(row["baseline_id"], f"{label}.baseline_id", maximum=160),
        reconstruction_id=reconstruction_id,
        source=_source(row["source"], f"{label}.source"),
        runtime_image_inspect_raw=_descriptor(
            row["runtime_image_inspect_raw"], f"{label}.runtime_image_inspect_raw"
        ),
        archive=_descriptor(row["oci_archive"], f"{label}.oci_archive"),
        layout=_descriptor(row["oci_layout"], f"{label}.oci_layout"),
        manifest=_descriptor(row["oci_manifest"], f"{label}.oci_manifest"),
        image_id=_image_id(row["image_id"], f"{label}.image_id"),
    )


def _equality_descriptor(value: Any, label: str) -> EqualityDescriptor:
    row = _exact(value, {"a", "b", "sha256"}, label)
    a = _descriptor(row["a"], f"{label}.a")
    b = _descriptor(row["b"], f"{label}.b")
    digest = _sha256(row["sha256"], f"{label}.sha256")
    if a.sha256 != b.sha256 or a.sha256 != digest or a.byte_length != b.byte_length:
        _fail("a-b-equality-mismatch", f"{label} must bind equal checksums and byte lengths")
    return EqualityDescriptor(a=a, b=b, sha256=digest)


def _equality(value: Any, label: str) -> Equality:
    row = _exact(
        value,
        {"binary", "bundle", "oci_archive", "oci_layout", "oci_manifest", "oci_image"},
        label,
    )
    image_row = _exact(row["oci_image"], {"a", "b", "image_id"}, f"{label}.oci_image")
    image_a = _image_id(image_row["a"], f"{label}.oci_image.a")
    image_b = _image_id(image_row["b"], f"{label}.oci_image.b")
    image_id = _image_id(image_row["image_id"], f"{label}.oci_image.image_id")
    if image_a != image_b or image_a != image_id:
        _fail("a-b-equality-mismatch", f"{label}.oci_image must bind one image ID")
    return Equality(
        binary=_equality_descriptor(row["binary"], f"{label}.binary"),
        bundle=_equality_descriptor(row["bundle"], f"{label}.bundle"),
        oci_archive=_equality_descriptor(row["oci_archive"], f"{label}.oci_archive"),
        oci_layout=_equality_descriptor(row["oci_layout"], f"{label}.oci_layout"),
        oci_manifest=_equality_descriptor(row["oci_manifest"], f"{label}.oci_manifest"),
        oci_image_id=image_id,
    )


def parse_manifest(value: dict[str, Any]) -> BaselineManifest:
    row = _exact(
        value,
        {
            "schema_version",
            "baseline_id",
            "baseline_kind",
            "provenance_class",
            "historical_distribution",
            "historical_stable_artifact_status",
            "was_previously_shipped",
            "source",
            "reproductions",
            "equality",
        },
        "reconstructed prior baseline",
    )
    if row["schema_version"] != MANIFEST_VERSION:
        _fail("unsupported-baseline-version", "baseline.schema_version is unsupported")
    if row["baseline_kind"] != BASELINE_KIND:
        _fail("non-reconstructed-baseline", f"baseline_kind must be {BASELINE_KIND}")
    if row["provenance_class"] != PROVENANCE_CLASS:
        _fail("invalid-provenance-class", f"provenance_class must be {PROVENANCE_CLASS}")
    if row["historical_distribution"] != HISTORICAL_DISTRIBUTION:
        _fail(
            "historical-distribution-claim",
            f"historical_distribution must be {HISTORICAL_DISTRIBUTION}",
        )
    if row["historical_stable_artifact_status"] != HISTORICAL_STABLE_ARTIFACT_STATUS:
        _fail(
            "historical-stable-artifact-claim",
            "historical_stable_artifact_status must remain unavailable",
        )
    if row["was_previously_shipped"] is not WAS_PREVIOUSLY_SHIPPED:
        _fail("historical-shipped-claim", "was_previously_shipped must be false")
    source = _source(row["source"], "baseline.source")
    baseline_id = _string(row["baseline_id"], "baseline.baseline_id", maximum=160)
    expected_baseline_id = f"reconstructed-{source.tag_name}"
    if baseline_id != expected_baseline_id:
        _fail("baseline-id-tag-mismatch", f"baseline_id must be {expected_baseline_id}")
    reproductions = _exact(row["reproductions"], {"a", "b"}, "baseline.reproductions")
    a_receipt = _descriptor(reproductions["a"], "baseline.reproductions.a")
    b_receipt = _descriptor(reproductions["b"], "baseline.reproductions.b")
    if a_receipt.path == b_receipt.path:
        _fail("non-independent-reconstruction", "A and B must use distinct receipt files")
    return BaselineManifest(
        baseline_id=baseline_id,
        source=source,
        a_receipt=a_receipt,
        b_receipt=b_receipt,
        equality=_equality(row["equality"], "baseline.equality"),
    )


def _verify_raw_descriptor(root_fd: int, descriptor: Descriptor, label: str) -> None:
    """Verify an artifact leaf through the common streaming FD reader."""

    common.verify_descriptor_file(root_fd, descriptor, label)


def _read_json_descriptor(root_fd: int, descriptor: Descriptor, label: str) -> dict[str, Any]:
    _raw, document = common.read_descriptor_json(root_fd, descriptor, label)
    return document


def _read_runtime_image_inspect_id(
    root_fd: int,
    descriptor: Descriptor,
    label: str,
) -> str:
    """Bind an original, noncanonical ``docker image inspect`` JSON capture.

    Docker emits an array containing one image object.  It is intentionally
    parsed with the common strict (but noncanonical) parser after descriptor
    length/hash verification, preserving the original tool bytes as evidence.
    """

    raw = common.read_descriptor_bytes(
        root_fd,
        descriptor,
        label,
        maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
    )
    document = common.parse_strict_json(
        raw,
        label,
        maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        require_object=False,
    )
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        _fail("invalid-runtime-image-inspect", f"{label} must be a one-image Docker inspect array")
    row = document[0]
    if "Id" not in row:
        _fail("invalid-runtime-image-inspect", f"{label}[0] must contain Docker Id")
    return _image_id(row["Id"], f"{label}[0].Id")


def _read_tag_identity(root_fd: int, source: SourceBinding) -> TagIdentity:
    object_row = _exact(
        _read_json_descriptor(root_fd, source.tag_object, "raw Git tag object"),
        {
            "schema_version",
            "tag_ref",
            "object_type",
            "object_sha1",
            "target_object_type",
            "target_object_sha1",
        },
        "raw Git tag object",
    )
    if object_row["schema_version"] != GIT_TAG_OBJECT_VERSION:
        _fail("unsupported-git-tag-object-version", "raw Git tag object version is unsupported")
    expected_ref = f"refs/tags/{source.tag_name}"
    if object_row["tag_ref"] != expected_ref:
        _fail("git-tag-ref-mismatch", "raw Git tag object names another tag")
    if object_row["object_type"] != "tag" or object_row["target_object_type"] != "commit":
        _fail("non-annotated-tag-object", "raw Git evidence must identify annotated tag -> commit")
    object_sha1 = _git_sha1(object_row["object_sha1"], "raw Git tag object.object_sha1")
    target_sha1 = _git_sha1(
        object_row["target_object_sha1"], "raw Git tag object.target_object_sha1"
    )
    target_row = _exact(
        _read_json_descriptor(root_fd, source.tag_target, "raw Git tag target"),
        {"schema_version", "tag_ref", "tag_object_sha1", "target_commit_sha1"},
        "raw Git tag target",
    )
    if target_row["schema_version"] != GIT_TAG_TARGET_VERSION:
        _fail("unsupported-git-tag-target-version", "raw Git tag target version is unsupported")
    if target_row["tag_ref"] != expected_ref:
        _fail("git-tag-ref-mismatch", "raw Git tag target names another tag")
    if _git_sha1(target_row["tag_object_sha1"], "raw Git tag target.tag_object_sha1") != object_sha1:
        _fail("git-tag-object-target-mismatch", "raw tag target does not bind the raw tag object")
    if _git_sha1(target_row["target_commit_sha1"], "raw Git tag target.target_commit_sha1") != target_sha1:
        _fail("git-tag-object-target-mismatch", "raw tag target does not bind tag target commit")
    return TagIdentity(
        tag_name=source.tag_name,
        tag_ref=expected_ref,
        tag_object_sha1=object_sha1,
        target_commit_sha1=target_sha1,
    )


def _assert_descriptor_equal(actual: Descriptor, declared: Descriptor, label: str) -> None:
    if actual != declared:
        _fail("descriptor-binding-mismatch", f"{label} does not bind its raw evidence leaf")


def _assert_source_equal(actual: SourceBinding, expected: SourceBinding, label: str) -> None:
    if actual != expected:
        _fail("source-binding-mismatch", f"{label} must bind the manifest Git/archive evidence")


def _assert_equality(
    expected_a: Descriptor,
    expected_b: Descriptor,
    declared: EqualityDescriptor,
    label: str,
) -> None:
    _assert_descriptor_equal(expected_a, declared.a, f"{label}.a")
    _assert_descriptor_equal(expected_b, declared.b, f"{label}.b")
    if expected_a.sha256 != expected_b.sha256 or expected_a.byte_length != expected_b.byte_length:
        _fail("a-b-equality-mismatch", f"{label} A/B raw descriptors disagree")


def _assert_independent_paths(manifest: BaselineManifest, a: BuildReceipt, b: BuildReceipt) -> None:
    a_paths = {
        a.recipe_inspect.path,
        a.image_inspect.path,
        a.runtime_image_inspect_raw.path,
        a.artifacts.binary.path,
        a.artifacts.bundle.path,
        a.artifacts.oci.archive.path,
        a.artifacts.oci.layout.path,
        a.artifacts.oci.manifest.path,
    }
    b_paths = {
        b.recipe_inspect.path,
        b.image_inspect.path,
        b.runtime_image_inspect_raw.path,
        b.artifacts.binary.path,
        b.artifacts.bundle.path,
        b.artifacts.oci.archive.path,
        b.artifacts.oci.layout.path,
        b.artifacts.oci.manifest.path,
    }
    if a_paths & b_paths:
        _fail("non-independent-reconstruction", "A and B must use distinct inspect/artifact paths")
    reserved = {
        manifest.source.tag_object.path,
        manifest.source.tag_target.path,
        manifest.source.archive.path,
        manifest.a_receipt.path,
        manifest.b_receipt.path,
    }
    if (a_paths | b_paths) & reserved:
        _fail("duplicate-evidence-path", "raw Git/archive/receipt leaves cannot be reused as outputs")


def _require_global_unique_descriptors(
    manifest: BaselineManifest,
    a: BuildReceipt,
    b: BuildReceipt,
    recipe_a: RecipeInspect,
    recipe_b: RecipeInspect,
) -> None:
    """Reject a path alias across every independently captured evidence leaf.

    Descriptor copies in receipts/inspect documents are intentionally not
    listed twice: their exact equality to the canonical instance is checked
    separately.  The list below instead represents each physical raw leaf
    exactly once, so a reused path cannot make two observations appear
    independent.
    """

    common.require_unique_descriptors(
        (
            manifest.source.tag_object,
            manifest.source.tag_target,
            manifest.source.archive,
            manifest.a_receipt,
            manifest.b_receipt,
            a.recipe_inspect,
            a.image_inspect,
            a.runtime_image_inspect_raw,
            recipe_a.recipe,
            a.artifacts.binary,
            a.artifacts.bundle,
            a.artifacts.oci.archive,
            a.artifacts.oci.layout,
            a.artifacts.oci.manifest,
            b.recipe_inspect,
            b.image_inspect,
            b.runtime_image_inspect_raw,
            recipe_b.recipe,
            b.artifacts.binary,
            b.artifacts.bundle,
            b.artifacts.oci.archive,
            b.artifacts.oci.layout,
            b.artifacts.oci.manifest,
        ),
        "baseline raw evidence",
    )


def _validate_inspects(
    root_fd: int,
    manifest: BaselineManifest,
    receipt: BuildReceipt,
    tag_identity: TagIdentity,
) -> tuple[RecipeInspect, ImageInspect]:
    recipe = _recipe_inspect(
        _read_json_descriptor(root_fd, receipt.recipe_inspect, f"reconstruction {receipt.reconstruction_id} recipe inspect"),
        f"reconstruction {receipt.reconstruction_id} recipe inspect",
    )
    image = _image_inspect(
        _read_json_descriptor(root_fd, receipt.image_inspect, f"reconstruction {receipt.reconstruction_id} image inspect"),
        f"reconstruction {receipt.reconstruction_id} image inspect",
    )
    for inspect, label in ((recipe, "recipe inspect"), (image, "image inspect")):
        if inspect.baseline_id != manifest.baseline_id:
            _fail("baseline-binding-mismatch", f"{label} belongs to another baseline")
        if inspect.reconstruction_id != receipt.reconstruction_id:
            _fail("reconstruction-id-mismatch", f"{label} belongs to another reconstruction")
        _assert_source_equal(inspect.source, manifest.source, label)
    _assert_descriptor_equal(recipe.binary, receipt.artifacts.binary, "recipe inspect.binary")
    _assert_descriptor_equal(recipe.bundle, receipt.artifacts.bundle, "recipe inspect.bundle")
    _assert_descriptor_equal(
        image.runtime_image_inspect_raw,
        receipt.runtime_image_inspect_raw,
        "image inspect.runtime_image_inspect_raw",
    )
    _assert_descriptor_equal(image.archive, receipt.artifacts.oci.archive, "image inspect.oci_archive")
    _assert_descriptor_equal(image.layout, receipt.artifacts.oci.layout, "image inspect.oci_layout")
    _assert_descriptor_equal(image.manifest, receipt.artifacts.oci.manifest, "image inspect.oci_manifest")
    # The tag identity is intentionally read from raw Git leaves before any
    # builder-provided recipe/image capture is accepted.
    if recipe.source.tag_name != tag_identity.tag_name or image.source.tag_name != tag_identity.tag_name:
        _fail("git-tag-binding-mismatch", "inspect source names a different raw Git tag")
    raw_image_id = _read_runtime_image_inspect_id(
        root_fd,
        receipt.runtime_image_inspect_raw,
        f"reconstruction {receipt.reconstruction_id} raw Docker image inspect",
    )
    if raw_image_id != image.image_id:
        _fail(
            "runtime-image-inspect-id-mismatch",
            "raw Docker image inspect Id does not bind the canonical image inspect",
        )
    _verify_raw_descriptor(
        root_fd, recipe.recipe, f"reconstruction {receipt.reconstruction_id} raw recipe"
    )
    return recipe, image


def _verify_artifact_leaves(root_fd: int, manifest: BaselineManifest, a: BuildReceipt, b: BuildReceipt) -> None:
    _verify_raw_descriptor(root_fd, manifest.source.archive, "source archive")
    for receipt in (a, b):
        prefix = f"reconstruction {receipt.reconstruction_id}"
        _verify_raw_descriptor(root_fd, receipt.artifacts.binary, f"{prefix} server binary")
        _verify_raw_descriptor(root_fd, receipt.artifacts.bundle, f"{prefix} bundle")
        _verify_raw_descriptor(root_fd, receipt.artifacts.oci.archive, f"{prefix} OCI archive")
        _verify_raw_descriptor(root_fd, receipt.artifacts.oci.layout, f"{prefix} OCI layout")
        _verify_raw_descriptor(root_fd, receipt.artifacts.oci.manifest, f"{prefix} OCI manifest")


def evaluate(root_fd: int, document: dict[str, Any]) -> dict[str, Any]:
    """Validate a canonical manifest using an already pinned evidence-root FD."""

    manifest = parse_manifest(document)
    tag_identity = _read_tag_identity(root_fd, manifest.source)
    a = _build_receipt(
        _read_json_descriptor(root_fd, manifest.a_receipt, "reconstruction A receipt"),
        "reconstruction A receipt",
    )
    b = _build_receipt(
        _read_json_descriptor(root_fd, manifest.b_receipt, "reconstruction B receipt"),
        "reconstruction B receipt",
    )
    if a.reconstruction_id != "a" or b.reconstruction_id != "b":
        _fail("reconstruction-id-mismatch", "A/B receipt paths must hold their matching IDs")
    for receipt in (a, b):
        if receipt.baseline_id != manifest.baseline_id:
            _fail("baseline-binding-mismatch", "build receipt belongs to another baseline")
        _assert_source_equal(receipt.source, manifest.source, "build receipt source")
    _assert_independent_paths(manifest, a, b)
    recipe_a, image_a = _validate_inspects(root_fd, manifest, a, tag_identity)
    recipe_b, image_b = _validate_inspects(root_fd, manifest, b, tag_identity)
    _require_global_unique_descriptors(manifest, a, b, recipe_a, recipe_b)
    _assert_equality(a.artifacts.binary, b.artifacts.binary, manifest.equality.binary, "server binary")
    _assert_equality(a.artifacts.bundle, b.artifacts.bundle, manifest.equality.bundle, "bundle")
    _assert_equality(
        a.artifacts.oci.archive,
        b.artifacts.oci.archive,
        manifest.equality.oci_archive,
        "OCI archive",
    )
    _assert_equality(a.artifacts.oci.layout, b.artifacts.oci.layout, manifest.equality.oci_layout, "OCI layout")
    _assert_equality(a.artifacts.oci.manifest, b.artifacts.oci.manifest, manifest.equality.oci_manifest, "OCI manifest")
    if image_a.image_id != image_b.image_id or image_a.image_id != manifest.equality.oci_image_id:
        _fail("a-b-equality-mismatch", "OCI image inspect leaves do not bind one exact image ID")
    _verify_artifact_leaves(root_fd, manifest, a, b)
    return {
        "schema_version": CHECK_REPORT_VERSION,
        "status": "passed",
        "passed": True,
        "baseline_id": manifest.baseline_id,
        "baseline_kind": BASELINE_KIND,
        "provenance_class": PROVENANCE_CLASS,
        "historical_distribution": HISTORICAL_DISTRIBUTION,
        "historical_stable_artifact_status": HISTORICAL_STABLE_ARTIFACT_STATUS,
        "was_previously_shipped": WAS_PREVIOUSLY_SHIPPED,
        "source_archive_content_binding": "not-validated",
        "oci_archive_content_binding": "not-validated",
        "source": manifest.source.as_json(),
        "git_identity": {
            "tag_name": tag_identity.tag_name,
            "tag_ref": tag_identity.tag_ref,
            "tag_object_sha1": tag_identity.tag_object_sha1,
            "target_commit_sha1": tag_identity.target_commit_sha1,
        },
        "reproductions": {
            "a": {
                "receipt": manifest.a_receipt.as_json(),
                "recipe_inspect": a.recipe_inspect.as_json(),
                "image_inspect": a.image_inspect.as_json(),
                "runtime_image_inspect_raw": a.runtime_image_inspect_raw.as_json(),
                "artifacts": a.artifacts.as_json(),
            },
            "b": {
                "receipt": manifest.b_receipt.as_json(),
                "recipe_inspect": b.recipe_inspect.as_json(),
                "image_inspect": b.image_inspect.as_json(),
                "runtime_image_inspect_raw": b.runtime_image_inspect_raw.as_json(),
                "artifacts": b.artifacts.as_json(),
            },
        },
        "equality": manifest.equality.as_json(),
        "checks": [{"name": name, "passed": True} for name in CHECK_NAMES],
        "reason_codes": [],
    }


def validate_file(evidence_root: Path, manifest_relative: str) -> dict[str, Any]:
    root_fd = common.open_private_evidence_directory(evidence_root, "evidence root")
    try:
        raw = common.read_bounded_regular_relative(
            root_fd,
            common.validate_relative_path(manifest_relative, "baseline manifest"),
            "baseline manifest",
        )
        document = common.parse_canonical_json(raw, "baseline manifest")
        assert isinstance(document, dict)
        return evaluate(root_fd, document)
    finally:
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True, help="absolute physical evidence directory")
    parser.add_argument("--manifest", required=True, help="relative canonical baseline manifest")
    parser.add_argument(
        "--report-name",
        help="optional create-only report leaf name in the evidence root; stdout when omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root_fd: int | None = None
    try:
        root_fd = common.open_private_evidence_directory(args.evidence_root, "evidence root")
        raw = common.read_bounded_regular_relative(
            root_fd,
            common.validate_relative_path(args.manifest, "baseline manifest"),
            "baseline manifest",
        )
        document = common.parse_canonical_json(raw, "baseline manifest")
        assert isinstance(document, dict)
        report = evaluate(root_fd, document)
        if args.report_name:
            common.write_create_only_json(root_fd, args.report_name, report, "baseline check report")
        else:
            sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
        return 0
    except common.ProvenanceV2Error as error:
        payload = {
            "schema_version": CHECK_REPORT_VERSION,
            "status": "failed",
            "passed": False,
            "reason_codes": [getattr(error, "reason_code", "invalid-baseline")],
            "error": str(error),
        }
        sys.stderr.buffer.write(common.canonical_json_bytes(payload) + b"\n")
        return 1
    finally:
        if root_fd is not None:
            os.close(root_fd)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
