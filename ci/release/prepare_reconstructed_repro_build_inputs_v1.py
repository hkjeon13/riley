#!/usr/bin/env python3
"""Prepare a private closure for the reviewed PR16 A/B reproducibility inputs.

This source-only tool snapshots two already-captured ``repro-build-*.tar``
archives and replays the reviewed PR16 checker over fresh private runtime
copies.  The source archive is re-bound to the separately reviewed RC2 source
inputs closure on every replay.  It never invokes Docker, a compiler, a GPU,
or a service, and it does not claim that either binary or bundle was assembled
into a runtime image.

The PR16 checker predates the C02 held-FD evidence boundary and consumes
``Path`` objects.  This adapter never passes it an external input pathname:
all external bytes are first snapshotted through no-follow descriptors, then
materialized into a fresh exact-0700 checker directory.  The checker therefore
only reopens copies created from held immutable input descriptors.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, NoReturn, Sequence

sys.dont_write_bytecode = True

import check_reproducible_build as reproducibility
import prepare_reconstructed_rc2_inputs_v1 as source_inputs
import provenance_v2_common as common


REPRO_BUILD_INPUTS_VERSION = "riley.reconstructed-repro-build-inputs.v1"
REPRO_BUILD_INPUTS_NAME = "reconstructed-repro-build-inputs.json"
REPRO_BUILDS_DIRECTORY_NAME = "repro-builds"
CAPTURE_SCOPE = "existing-pr16-a-b-reproducibility-evidence-inputs"
PLATFORM = {"os": "linux", "architecture": "amd64"}
RECONSTRUCTION_IDS = ("a", "b")
BUILD_IDS = {"a": "A", "b": "B"}
SOURCE_LEAF_NAME = source_inputs.SOURCE_ARCHIVE_NAME

MAX_RECEIPT_BYTES = common.DEFAULT_MAX_JSON_BYTES
MAX_SOURCE_ARCHIVE_BYTES = reproducibility.MAX_SOURCE_ARCHIVE_SIZE
MAX_REPRO_ARCHIVE_BYTES = reproducibility.MAX_EVIDENCE_ARCHIVE_SIZE
MAX_BINARY_BYTES = reproducibility.MAX_BINARY_SIZE
MAX_BUNDLE_BYTES = reproducibility.MAX_EVIDENCE_MEMBER_SIZE
MAX_BUILD_MANIFEST_BYTES = reproducibility.MAX_METADATA_SIZE

BUILD_LEAVES = (
    ("evidence_archive", "repro-build-{arm}.tar", None, MAX_REPRO_ARCHIVE_BYTES),
    ("build_manifest", "build.json", "manifest/build.json", MAX_BUILD_MANIFEST_BYTES),
    ("binary", "riley", "bin/riley", MAX_BINARY_BYTES),
    ("bundle", "riley.tar.gz", "bundle/riley.tar.gz", MAX_BUNDLE_BYTES),
)

BINDING_STATUS = {
    "source_inputs": "replayed-reviewed-rc2-source-inputs-v1",
    "reproducibility_evidence": "replayed-pr16-release-build-reproducibility-v1",
    "source_archive": "validated",
    "binary_a_b": "validated",
    "bundle_a_b": "validated",
    "execution_independence": "validated",
}
NOT_ESTABLISHED = {
    "runtime_image_assembly": "not-established",
    "runtime_image_capture": "not-established",
    "source_to_runtime_image": "not-established",
    "bundle_to_runtime_image": "not-established",
    "oci_content": "not-established",
    "rollback": "not-established",
    "freeze": "not-established",
    "qualification": "not-run",
}


class ReproBuildInputsError(common.ProvenanceV2Error):
    """The PR16 reproducibility inputs cannot be prepared or replayed."""


def _fail(code: str, message: str) -> NoReturn:
    error = ReproBuildInputsError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Any) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _source(call: Any) -> Any:
    try:
        return call()
    except source_inputs.ReconstructedRc2InputsError as error:
        _fail(getattr(error, "reason_code", "invalid-source-inputs"), str(error))
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "invalid-source-inputs"), str(error))


def _reproducibility(call: Any) -> Any:
    try:
        return call()
    except reproducibility.ReleaseContractError as error:
        _fail("invalid-pr16-reproducibility-evidence", str(error))


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


def _expected_sha256(value: Any, label: str) -> str:
    if type(value) is not str or common.SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail("invalid-expected-sha256", f"{label} must be a non-zero lowercase SHA-256")
    return value


def _expected_build_image_id(value: Any) -> str:
    if type(value) is not str or reproducibility.IMAGE_ID_PATTERN.fullmatch(value) is None:
        _fail("invalid-expected-build-image-id", "--expected-build-image-id must be a non-zero sha256 image ID")
    if value == "sha256:" + "0" * 64:
        _fail("invalid-expected-build-image-id", "--expected-build-image-id must be non-zero")
    return value


def _exact(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else []
        _fail(
            "unknown-or-missing-field",
            f"{label} fields differ; expected={sorted(expected)}, actual={actual}",
        )
    return value


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


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _require_external_layout(
    evidence_root: Path,
    source_input_root: Path,
    repro_build_a: Path,
    repro_build_b: Path,
) -> None:
    if _paths_overlap(evidence_root, source_input_root):
        _fail("output-source-overlap", "--evidence-root must not overlap --source-input-root")
    for source, label in ((repro_build_a, "--repro-build-a"), (repro_build_b, "--repro-build-b")):
        if evidence_root == source or evidence_root in source.parents:
            _fail("output-input-overlap", f"--evidence-root must not contain {label}")
    if repro_build_a == repro_build_b:
        _fail("input-alias", "--repro-build-a and --repro-build-b must be distinct files")
    try:
        a_metadata = os.lstat(repro_build_a)
        b_metadata = os.lstat(repro_build_b)
    except OSError as error:
        _fail("missing-input", f"cannot inspect reproducibility inputs before output creation: {error}")
    if (a_metadata.st_dev, a_metadata.st_ino) == (b_metadata.st_dev, b_metadata.st_ino):
        _fail("input-alias", "--repro-build-a and --repro-build-b must not share an inode")


def _source_projection(
    source_row: Mapping[str, Any],
    source_receipt: common.EvidenceDescriptor,
    expected_source_archive_sha256: str,
) -> dict[str, Any]:
    row = _exact(
        source_row,
        {
            "schema_version",
            "status",
            "qualification_status",
            "baseline_id",
            "source",
            "git_identity",
            "expected_source_archive_sha256",
            "archive_generation",
        },
        "reviewed source inputs receipt",
    )
    if row["schema_version"] != source_inputs.SOURCE_INPUTS_VERSION:
        _fail("invalid-source-inputs", "reviewed source receipt has an unexpected schema version")
    if row["baseline_id"] != source_inputs.RECONSTRUCTED_RC2_BASELINE_ID:
        _fail("invalid-source-inputs", "reviewed source receipt has an unexpected baseline ID")
    if row["expected_source_archive_sha256"] != expected_source_archive_sha256:
        _fail("reviewed-source-archive-digest-mismatch", "reviewed source receipt differs from caller SHA")
    source = _exact(row["source"], {"tag_name", "tag_object", "tag_target", "archive"}, "reviewed source")
    if source["tag_name"] != source_inputs.RECONSTRUCTED_RC2_TAG:
        _fail("invalid-source-inputs", "reviewed source receipt has an unexpected tag name")
    for field, maximum in (("tag_object", MAX_RECEIPT_BYTES), ("tag_target", MAX_RECEIPT_BYTES), ("archive", MAX_SOURCE_ARCHIVE_BYTES)):
        descriptor = _common(lambda field=field: common.parse_descriptor(source[field], f"reviewed source.{field}"))
        if descriptor.byte_length > maximum:
            _fail("invalid-source-inputs", f"reviewed source.{field} exceeds its byte bound")
    identity = _exact(
        row["git_identity"],
        {"tag_ref", "tag_object_sha1", "target_commit_sha1"},
        "reviewed source git_identity",
    )
    if identity != {
        "tag_ref": source_inputs.RECONSTRUCTED_RC2_TAG_REF,
        "tag_object_sha1": source_inputs.RECONSTRUCTED_RC2_TAG_OBJECT,
        "target_commit_sha1": source_inputs.RECONSTRUCTED_RC2_TARGET,
    }:
        _fail("invalid-source-inputs", "reviewed source receipt does not retain the pinned RC2 identity")
    return {
        "receipt": source_receipt.as_json(),
        "expected_source_archive_sha256": expected_source_archive_sha256,
        "git_identity": dict(identity),
        "source": dict(source),
    }


def _descriptor(value: Any, label: str, expected_path: str, maximum_bytes: int) -> common.EvidenceDescriptor:
    descriptor = _common(lambda: common.parse_descriptor(value, label))
    if descriptor.path != expected_path:
        _fail("evidence-path-mismatch", f"{label} must use the fixed path {expected_path!r}")
    if descriptor.byte_length < 1 or descriptor.byte_length > maximum_bytes:
        _fail("invalid-evidence-descriptor", f"{label} has an invalid byte length")
    return descriptor


def _parse_receipt(receipt: Any) -> dict[str, Any]:
    row = _exact(
        receipt,
        {
            "schema_version",
            "status",
            "qualification_status",
            "capture_scope",
            "baseline_id",
            "source_inputs",
            "reproducibility_contract",
            "builds",
            "equality",
            "binding_status",
            "not_established",
        },
        "reproducibility inputs receipt",
    )
    if (
        row["schema_version"] != REPRO_BUILD_INPUTS_VERSION
        or row["status"] != "prepared"
        or row["qualification_status"] != "not-run"
        or row["capture_scope"] != CAPTURE_SCOPE
        or row["baseline_id"] != source_inputs.RECONSTRUCTED_RC2_BASELINE_ID
    ):
        _fail("invalid-repro-build-inputs-receipt", "receipt does not retain the exact v1 scalars")
    source = _exact(
        row["source_inputs"],
        {"receipt", "expected_source_archive_sha256", "git_identity", "source"},
        "receipt source_inputs",
    )
    _descriptor(source["receipt"], "receipt source_inputs.receipt", source_inputs.SOURCE_INPUTS_NAME, MAX_RECEIPT_BYTES)
    _expected_sha256(source["expected_source_archive_sha256"], "receipt source_inputs.expected_source_archive_sha256")
    _exact(source["git_identity"], {"tag_ref", "tag_object_sha1", "target_commit_sha1"}, "receipt source_inputs.git_identity")
    source_binding = _exact(source["source"], {"tag_name", "tag_object", "tag_target", "archive"}, "receipt source_inputs.source")
    _descriptor(source_binding["tag_object"], "receipt source tag_object", "source/git-tag-object.json", MAX_RECEIPT_BYTES)
    _descriptor(source_binding["tag_target"], "receipt source tag_target", "source/git-tag-target.json", MAX_RECEIPT_BYTES)
    _descriptor(source_binding["archive"], "receipt source archive", f"source/{SOURCE_LEAF_NAME}", MAX_SOURCE_ARCHIVE_BYTES)
    contract = _exact(
        row["reproducibility_contract"],
        {"schema_version", "gate_id", "source_revision", "source_date_epoch", "build_image_id", "platform", "network", "independent_clean_containers"},
        "receipt reproducibility_contract",
    )
    if (
        contract["schema_version"] != reproducibility.SCHEMA_VERSION
        or contract["gate_id"] != reproducibility.GATE_ID
        or type(contract["source_revision"]) is not str
        or reproducibility.REVISION_PATTERN.fullmatch(contract["source_revision"]) is None
        or type(contract["source_date_epoch"]) is not int
        or isinstance(contract["source_date_epoch"], bool)
        or not 0 <= contract["source_date_epoch"] <= 0xFFFFFFFF
        or _expected_build_image_id(contract["build_image_id"]) != contract["build_image_id"]
        or contract["platform"] != PLATFORM
        or contract["network"] != "none"
        or contract["independent_clean_containers"] != 2
    ):
        _fail("invalid-repro-build-inputs-receipt", "receipt reproducibility contract is invalid")
    builds = _exact(row["builds"], set(RECONSTRUCTION_IDS), "receipt builds")
    parsed_builds: dict[str, dict[str, common.EvidenceDescriptor]] = {}
    for arm in RECONSTRUCTION_IDS:
        build = _exact(
            builds[arm],
            {"reconstruction_id", "evidence_build_id", "evidence_archive", "build_manifest", "binary", "bundle"},
            f"receipt builds.{arm}",
        )
        if build["reconstruction_id"] != arm or build["evidence_build_id"] != BUILD_IDS[arm]:
            _fail("invalid-repro-build-inputs-receipt", f"receipt build {arm} has an invalid arm identity")
        descriptors: dict[str, common.EvidenceDescriptor] = {}
        for field, leaf, _member, maximum in BUILD_LEAVES:
            descriptors[field] = _descriptor(
                build[field],
                f"receipt builds.{arm}.{field}",
                f"{REPRO_BUILDS_DIRECTORY_NAME}/{arm}/{leaf.format(arm=arm)}",
                maximum,
            )
        parsed_builds[arm] = descriptors
    _common(lambda: common.require_unique_descriptors(tuple(item for build in parsed_builds.values() for item in build.values()), "receipt build descriptors"))
    equality = _exact(row["equality"], {"binary", "bundle"}, "receipt equality")
    for field in ("binary", "bundle"):
        pair = _exact(equality[field], {"a", "b", "sha256"}, f"receipt equality.{field}")
        if pair["sha256"] != parsed_builds["a"][field].sha256 or pair["sha256"] != parsed_builds["b"][field].sha256:
            _fail("invalid-repro-build-inputs-receipt", f"receipt equality.{field} SHA-256 does not match both arms")
        for arm in RECONSTRUCTION_IDS:
            descriptor = _descriptor(
                pair[arm],
                f"receipt equality.{field}.{arm}",
                parsed_builds[arm][field].path,
                MAX_BINARY_BYTES if field == "binary" else MAX_BUNDLE_BYTES,
            )
            if descriptor != parsed_builds[arm][field]:
                _fail("invalid-repro-build-inputs-receipt", f"receipt equality.{field}.{arm} differs from its build descriptor")
    if row["binding_status"] != BINDING_STATUS or row["not_established"] != NOT_ESTABLISHED:
        _fail("invalid-repro-build-inputs-receipt", "receipt makes an unsupported binding claim")
    return {**row, "_build_descriptors": parsed_builds}


@contextmanager
def _private_checker_directory() -> Iterator[tuple[Path, int]]:
    with tempfile.TemporaryDirectory(prefix="riley-repro-inputs-v1-") as temporary:
        path = Path(temporary)
        try:
            os.chmod(path, 0o700)
        except OSError as error:
            _fail("unsafe-checker-directory", f"cannot make private checker directory exact-0700: {error}")
        descriptor = _common(lambda: common.open_private_evidence_directory(path, "private checker directory"))
        try:
            yield path, descriptor
        finally:
            os.close(descriptor)


def _member_digest(archive_path: Path, arm: str, relative: str, maximum_bytes: int) -> tuple[str, int]:
    name = f"riley-repro-build-{arm}/{relative}"
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            member = archive.getmember(name)
            if not member.isreg() or member.size < 1 or member.size > maximum_bytes:
                _fail("invalid-pr16-reproducibility-evidence", f"selected evidence member is invalid: {name}")
            source = archive.extractfile(member)
            if source is None:
                _fail("invalid-pr16-reproducibility-evidence", f"cannot read selected evidence member: {name}")
            digest = hashlib.sha256()
            remaining = member.size
            while remaining:
                chunk = source.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
                if not chunk:
                    _fail("invalid-pr16-reproducibility-evidence", f"selected evidence member truncated: {name}")
                digest.update(chunk)
                remaining -= len(chunk)
            if source.read(1):
                _fail("invalid-pr16-reproducibility-evidence", f"selected evidence member grew: {name}")
    except (tarfile.TarError, KeyError) as error:
        _fail("invalid-pr16-reproducibility-evidence", f"cannot parse selected evidence member {name}: {error}")
    return digest.hexdigest(), member.size


def _extract_member_to_temporary(
    archive_path: Path,
    arm: str,
    relative: str,
    maximum_bytes: int,
    directory: Path,
) -> Path:
    name = f"riley-repro-build-{arm}/{relative}"
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            member = archive.getmember(name)
            if not member.isreg() or member.size < 1 or member.size > maximum_bytes:
                _fail("invalid-pr16-reproducibility-evidence", f"selected evidence member is invalid: {name}")
            source = archive.extractfile(member)
            if source is None:
                _fail("invalid-pr16-reproducibility-evidence", f"cannot read selected evidence member: {name}")
            with tempfile.NamedTemporaryFile(prefix="selected-", dir=directory, delete=False) as output:
                path = Path(output.name)
                remaining = member.size
                while remaining:
                    chunk = source.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
                    if not chunk:
                        _fail("invalid-pr16-reproducibility-evidence", f"selected evidence member truncated: {name}")
                    output.write(chunk)
                    remaining -= len(chunk)
                if source.read(1):
                    _fail("invalid-pr16-reproducibility-evidence", f"selected evidence member grew: {name}")
                output.flush()
                os.fsync(output.fileno())
    except (tarfile.TarError, KeyError, OSError) as error:
        _fail("invalid-pr16-reproducibility-evidence", f"cannot extract selected evidence member {name}: {error}")
    try:
        os.chmod(path, 0o600)
    except OSError as error:
        _fail("unsafe-checker-directory", f"cannot make selected temporary input private: {error}")
    return path


def _with_materialized_replay(
    *,
    source_root_fd: int,
    source_archive: common.EvidenceDescriptor,
    arm_descriptors: Mapping[str, Mapping[str, common.EvidenceDescriptor]],
    arm_fds: Mapping[str, int],
    expected_source_archive_sha256: str,
    source_revision: str,
    expected_build_image_id: str,
    consumer: Any,
) -> Any:
    """Keep trusted checker copies alive for one non-I/O callback."""

    with _private_checker_directory() as (checker_path, checker_fd):
        _common(
            lambda: common.materialize_descriptor_runtime_copy(
                source_root_fd,
                source_archive,
                checker_fd,
                "source.tar",
                "reviewed source archive checker copy",
                maximum_bytes=MAX_SOURCE_ARCHIVE_BYTES,
            )
        )
        source_path = checker_path / "source.tar"
        source_date_epoch = _reproducibility(
            lambda: reproducibility.derive_source_date_epoch(source_path, source_revision)
        )
        archives: dict[str, Path] = {}
        for arm in RECONSTRUCTION_IDS:
            leaf = f"repro-build-{arm}.tar"
            _common(
                lambda arm=arm, leaf=leaf: common.materialize_descriptor_runtime_copy(
                    arm_fds[arm],
                    arm_descriptors[arm]["evidence_archive"],
                    checker_fd,
                    leaf,
                    f"reproducibility evidence {arm} checker copy",
                    maximum_bytes=MAX_REPRO_ARCHIVE_BYTES,
                )
            )
            archives[arm] = checker_path / leaf
        facts = _reproducibility(
            lambda: reproducibility.validate_reproducibility_inputs(
                evidence_a=archives["a"],
                evidence_b=archives["b"],
                source_archive=source_path,
                expected_source_archive_sha256=expected_source_archive_sha256,
                source_revision=source_revision,
                source_date_epoch=source_date_epoch,
                build_image_id=expected_build_image_id,
            )
        )
        return consumer(checker_path, source_path, archives, facts)


def _verify_replay_facts(
    parsed: Mapping[str, Any],
    facts: Mapping[str, Any],
    archives: Mapping[str, Path],
) -> None:
    contract = parsed["reproducibility_contract"]
    source = facts["source"]
    if source != {
        "revision": contract["source_revision"],
        "archive_sha256": parsed["source_inputs"]["expected_source_archive_sha256"],
        "source_date_epoch": contract["source_date_epoch"],
    }:
        _fail("replay-fact-mismatch", "replayed source facts differ from the receipt")
    if facts["build"] != {
        "image_id": contract["build_image_id"],
        "image_inspect_sha256": facts["build"]["image_inspect_sha256"],
        "platform": reproducibility.PLATFORM,
        "network": "none",
        "independent_clean_containers": 2,
    }:
        _fail("replay-fact-mismatch", "replayed PR16 build facts differ from the receipt contract")
    descriptors: Mapping[str, Mapping[str, common.EvidenceDescriptor]] = parsed["_build_descriptors"]
    for arm in RECONSTRUCTION_IDS:
        replay = facts["reproductions"][arm]
        archive = descriptors[arm]["evidence_archive"]
        if replay["evidence_archive_sha256"] != archive.sha256:
            _fail("replay-fact-mismatch", f"replayed evidence archive SHA differs for arm {arm}")
        for field, _leaf, member, maximum in BUILD_LEAVES:
            descriptor = descriptors[arm][field]
            if field == "evidence_archive":
                continue
            assert member is not None
            digest, length = _member_digest(archives[arm], arm, member, maximum)
            if digest != descriptor.sha256 or length != descriptor.byte_length:
                _fail("selected-member-mismatch", f"selected {field} leaf differs from arm {arm} raw evidence")
            if field in {"binary", "bundle"}:
                artifact = replay["artifacts"][field]
                if artifact != {"sha256": descriptor.sha256, "byte_length": descriptor.byte_length}:
                    _fail("replay-fact-mismatch", f"replayed {field} facts differ for arm {arm}")
    for field in ("binary", "bundle"):
        left = descriptors["a"][field]
        right = descriptors["b"][field]
        if left.sha256 != right.sha256 or left.byte_length != right.byte_length:
            _fail("replay-fact-mismatch", f"replayed {field} is not byte-exact across A/B")


def _open_builds_for_replay(
    root_fd: int,
    descriptors: Mapping[str, Mapping[str, common.EvidenceDescriptor]],
) -> tuple[int, dict[str, int], dict[str, dict[str, common.EvidenceDescriptor]]]:
    repro_fd = _common(
        lambda: common.open_private_child_directory(root_fd, REPRO_BUILDS_DIRECTORY_NAME, "reproducibility evidence directory")
    )
    try:
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd, repro_fd, REPRO_BUILDS_DIRECTORY_NAME, "held reproducibility evidence directory"
            )
        )
        _assert_entries(repro_fd, set(RECONSTRUCTION_IDS), "reproducibility evidence directory")
        arm_fds: dict[str, int] = {}
        held: dict[str, dict[str, common.EvidenceDescriptor]] = {}
        for arm in RECONSTRUCTION_IDS:
            arm_fd = _common(
                lambda arm=arm: common.open_private_child_directory(repro_fd, arm, f"reproducibility evidence arm {arm}")
            )
            _common(
                lambda arm=arm, arm_fd=arm_fd: common.require_private_child_directory_fd(
                    repro_fd, arm_fd, arm, f"held reproducibility evidence arm {arm}"
                )
            )
            expected_entries = {leaf.format(arm=arm) for _field, leaf, _member, _maximum in BUILD_LEAVES}
            _assert_entries(arm_fd, expected_entries, f"reproducibility evidence arm {arm}")
            arm_descriptors: dict[str, common.EvidenceDescriptor] = {}
            for field, leaf, _member, maximum in BUILD_LEAVES:
                original = descriptors[arm][field]
                rebased = _common(
                    lambda original=original, leaf=leaf, arm=arm, field=field: common.rebase_descriptor_to_held_leaf(
                        original,
                        expected_root_relative_path=f"{REPRO_BUILDS_DIRECTORY_NAME}/{arm}/{leaf.format(arm=arm)}",
                        leaf_name=leaf.format(arm=arm),
                        label=f"reproducibility arm {arm} {field}",
                    )
                )
                _common(
                    lambda rebased=rebased, field=field, maximum=maximum, arm_fd=arm_fd, arm=arm: common.verify_private_snapshot_descriptor_file(
                        arm_fd,
                        rebased,
                        f"reproducibility arm {arm} {field}",
                        maximum_bytes=maximum,
                    )
                )
                arm_descriptors[field] = rebased
            arm_fds[arm] = arm_fd
            held[arm] = arm_descriptors
        return repro_fd, arm_fds, held
    except BaseException:
        for descriptor in locals().get("arm_fds", {}).values():
            os.close(descriptor)
        os.close(repro_fd)
        raise


def _verify_reconstructed_repro_build_inputs_fd(
    root_fd: int,
    source_root_fd: int,
    *,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
) -> dict[str, Any]:
    expected_source_sha = _expected_sha256(expected_source_archive_sha256, "--expected-source-archive-sha256")
    expected_image = _expected_build_image_id(expected_build_image_id)
    _common(lambda: common.require_private_evidence_directory_fd(root_fd, "reproducibility inputs root"))
    _common(lambda: common.require_private_evidence_directory_fd(source_root_fd, "RC2 source inputs root"))
    if os.fstat(root_fd).st_dev == os.fstat(source_root_fd).st_dev and os.fstat(root_fd).st_ino == os.fstat(source_root_fd).st_ino:
        _fail("root-inode-alias", "reproducibility inputs root must not alias the source inputs root")
    source_row = _source(
        lambda: source_inputs.verify_reconstructed_rc2_inputs_fd(
            source_root_fd,
            expected_source_archive_sha256=expected_source_sha,
        )
    )
    source_receipt = _common(
        lambda: common.describe_regular_relative(
            source_root_fd,
            source_inputs.SOURCE_INPUTS_NAME,
            "reviewed source inputs receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    source_projection = _source_projection(source_row, source_receipt, expected_source_sha)
    _assert_entries(root_fd, {REPRO_BUILDS_DIRECTORY_NAME, REPRO_BUILD_INPUTS_NAME}, "reproducibility inputs root")
    receipt = _common(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            REPRO_BUILD_INPUTS_NAME,
            "reproducibility inputs receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    parsed = _parse_receipt(receipt)
    if parsed["source_inputs"] != source_projection:
        _fail("source-inputs-binding-mismatch", "receipt source input binding differs from held reviewed source inputs")
    contract = parsed["reproducibility_contract"]
    if contract["source_revision"] != source_projection["git_identity"]["target_commit_sha1"]:
        _fail("source-inputs-binding-mismatch", "receipt source revision differs from the reviewed source target")
    if contract["build_image_id"] != expected_image:
        _fail("reviewed-build-image-id-mismatch", "receipt build image differs from caller-reviewed image ID")
    source_archive = _common(
        lambda: common.parse_descriptor(source_projection["source"]["archive"], "reviewed source archive")
    )
    repro_fd, arm_fds, held = _open_builds_for_replay(root_fd, parsed["_build_descriptors"])
    try:
        def replay(_checker_root: Path, _source_path: Path, archives: Mapping[str, Path], facts: Mapping[str, Any]) -> None:
            _verify_replay_facts(parsed, facts, archives)

        _with_materialized_replay(
            source_root_fd=source_root_fd,
            source_archive=source_archive,
            arm_descriptors=held,
            arm_fds=arm_fds,
            expected_source_archive_sha256=expected_source_sha,
            source_revision=contract["source_revision"],
            expected_build_image_id=expected_image,
            consumer=replay,
        )
        _assert_entries(repro_fd, set(RECONSTRUCTION_IDS), "reproducibility evidence directory")
        for arm in RECONSTRUCTION_IDS:
            expected_entries = {leaf.format(arm=arm) for _field, leaf, _member, _maximum in BUILD_LEAVES}
            _assert_entries(arm_fds[arm], expected_entries, f"reproducibility evidence arm {arm}")
            _common(
                lambda arm=arm: common.require_private_child_directory_fd(
                    repro_fd, arm_fds[arm], arm, f"held reproducibility evidence arm {arm}"
                )
            )
    finally:
        for arm_fd in arm_fds.values():
            os.close(arm_fd)
        os.close(repro_fd)
    _assert_entries(root_fd, {REPRO_BUILDS_DIRECTORY_NAME, REPRO_BUILD_INPUTS_NAME}, "reproducibility inputs root")
    terminal_receipt = _common(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            REPRO_BUILD_INPUTS_NAME,
            "reproducibility inputs receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    if terminal_receipt != receipt:
        _fail("raced-input", "reproducibility inputs receipt changed during replay")
    return {key: value for key, value in parsed.items() if key != "_build_descriptors"}


def verify_reconstructed_repro_build_inputs_fd(
    root_fd: int,
    source_root_fd: int,
    *,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
) -> dict[str, Any]:
    """Replay a pair closure using caller-held private root descriptors."""

    return _verify_reconstructed_repro_build_inputs_fd(
        root_fd,
        source_root_fd,
        expected_source_archive_sha256=expected_source_archive_sha256,
        expected_build_image_id=expected_build_image_id,
    )


def verify_reconstructed_repro_build_inputs(
    evidence_root: Path,
    *,
    source_input_root: Path,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
) -> dict[str, Any]:
    evidence_root = _normalized_absolute_path(evidence_root, "reproducibility inputs root")
    source_input_root = _normalized_absolute_path(source_input_root, "source inputs root")
    if _paths_overlap(evidence_root, source_input_root):
        _fail("output-source-overlap", "reproducibility inputs root must not overlap source inputs root")
    root_fd = _common(lambda: common.open_private_evidence_directory(evidence_root, "reproducibility inputs root"))
    source_root_fd = _common(lambda: common.open_private_evidence_directory(source_input_root, "RC2 source inputs root"))
    try:
        return verify_reconstructed_repro_build_inputs_fd(
            root_fd,
            source_root_fd,
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
        )
    finally:
        os.close(source_root_fd)
        os.close(root_fd)


def prepare_reconstructed_repro_build_inputs(
    evidence_root: Path,
    *,
    source_input_root: Path,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
    repro_build_a: Path,
    repro_build_b: Path,
) -> dict[str, Any]:
    """Create and self-replay one fresh A/B PR16 input closure."""

    evidence_root = _normalized_absolute_path(evidence_root, "--evidence-root")
    source_input_root = _normalized_absolute_path(source_input_root, "--source-input-root")
    repro_build_a = _normalized_absolute_path(repro_build_a, "--repro-build-a")
    repro_build_b = _normalized_absolute_path(repro_build_b, "--repro-build-b")
    expected_source_sha = _expected_sha256(expected_source_archive_sha256, "--expected-source-archive-sha256")
    expected_image = _expected_build_image_id(expected_build_image_id)
    _require_external_layout(evidence_root, source_input_root, repro_build_a, repro_build_b)
    source_root_fd = _common(lambda: common.open_private_evidence_directory(source_input_root, "RC2 source inputs root"))
    root_fd: int | None = None
    repro_fd: int | None = None
    arm_fds: dict[str, int] = {}
    try:
        source_row = _source(
            lambda: source_inputs.verify_reconstructed_rc2_inputs_fd(
                source_root_fd,
                expected_source_archive_sha256=expected_source_sha,
            )
        )
        source_receipt = _common(
            lambda: common.describe_regular_relative(
                source_root_fd,
                source_inputs.SOURCE_INPUTS_NAME,
                "reviewed source inputs receipt",
                maximum_bytes=MAX_RECEIPT_BYTES,
            )
        )
        source_projection = _source_projection(source_row, source_receipt, expected_source_sha)
        source_archive = _common(
            lambda: common.parse_descriptor(source_projection["source"]["archive"], "reviewed source archive")
        )
        root_fd = _common(lambda: common.create_private_evidence_directory(evidence_root, "reproducibility inputs root"))
        repro_fd = _common(
            lambda: common.create_private_child_directory(root_fd, REPRO_BUILDS_DIRECTORY_NAME, "reproducibility evidence directory")
        )
        descriptors: dict[str, dict[str, common.EvidenceDescriptor]] = {}
        for arm, input_path in (("a", repro_build_a), ("b", repro_build_b)):
            arm_fd = _common(
                lambda arm=arm: common.create_private_child_directory(repro_fd, arm, f"reproducibility evidence arm {arm}")
            )
            arm_fds[arm] = arm_fd
            archive_leaf = f"repro-build-{arm}.tar"
            snapshot = _common(
                lambda input_path=input_path, arm_fd=arm_fd, archive_leaf=archive_leaf, arm=arm: common.snapshot_absolute_regular_create_only(
                    input_path,
                    arm_fd,
                    archive_leaf,
                    f"raw reproducibility evidence {arm}",
                    maximum_bytes=MAX_REPRO_ARCHIVE_BYTES,
                    minimum_bytes=1,
                )
            )
            descriptors[arm] = {
                "evidence_archive": snapshot.descriptor(
                    f"{REPRO_BUILDS_DIRECTORY_NAME}/{arm}/{archive_leaf}",
                    f"raw reproducibility evidence {arm}",
                )
            }

        contract: dict[str, Any] = {}
        held_archives = {
            arm: {
                "evidence_archive": _common(
                    lambda arm=arm: common.rebase_descriptor_to_held_leaf(
                        descriptors[arm]["evidence_archive"],
                        expected_root_relative_path=(
                            f"{REPRO_BUILDS_DIRECTORY_NAME}/{arm}/repro-build-{arm}.tar"
                        ),
                        leaf_name=f"repro-build-{arm}.tar",
                        label=f"reproducibility evidence {arm} archive",
                    )
                )
            }
            for arm in RECONSTRUCTION_IDS
        }

        def capture_selected(checker_root: Path, _source_path: Path, archives: Mapping[str, Path], facts: Mapping[str, Any]) -> None:
            nonlocal contract
            source_facts = facts["source"]
            contract = {
                "schema_version": reproducibility.SCHEMA_VERSION,
                "gate_id": reproducibility.GATE_ID,
                "source_revision": source_facts["revision"],
                "source_date_epoch": source_facts["source_date_epoch"],
                "build_image_id": expected_image,
                "platform": dict(PLATFORM),
                "network": "none",
                "independent_clean_containers": 2,
            }
            for arm in RECONSTRUCTION_IDS:
                for field, leaf, member, maximum in BUILD_LEAVES:
                    if field == "evidence_archive":
                        continue
                    assert member is not None
                    temporary = _extract_member_to_temporary(archives[arm], arm, member, maximum, checker_root)
                    snapshot = _common(
                        lambda temporary=temporary, arm=arm, leaf=leaf, field=field, maximum=maximum: common.snapshot_absolute_regular_create_only(
                            temporary,
                            arm_fds[arm],
                            leaf.format(arm=arm),
                            f"reproducibility evidence {arm} selected {field}",
                            maximum_bytes=maximum,
                            minimum_bytes=1,
                        )
                    )
                    descriptor = snapshot.descriptor(
                        f"{REPRO_BUILDS_DIRECTORY_NAME}/{arm}/{leaf.format(arm=arm)}",
                        f"reproducibility evidence {arm} selected {field}",
                    )
                    if field in {"binary", "bundle"}:
                        artifact = facts["reproductions"][arm]["artifacts"][field]
                        if artifact != {"sha256": descriptor.sha256, "byte_length": descriptor.byte_length}:
                            _fail("selected-member-mismatch", f"selected {field} differs from validated arm {arm} evidence")
                    descriptors[arm][field] = descriptor

        _with_materialized_replay(
            source_root_fd=source_root_fd,
            source_archive=source_archive,
            arm_descriptors=held_archives,
            arm_fds=arm_fds,
            expected_source_archive_sha256=expected_source_sha,
            source_revision=source_projection["git_identity"]["target_commit_sha1"],
            expected_build_image_id=expected_image,
            consumer=capture_selected,
        )
        if contract["source_revision"] != source_projection["git_identity"]["target_commit_sha1"]:
            _fail("source-inputs-binding-mismatch", "validated PR16 source revision differs from reviewed RC2 source")
        for arm in RECONSTRUCTION_IDS:
            expected_entries = {leaf.format(arm=arm) for _field, leaf, _member, _maximum in BUILD_LEAVES}
            _assert_entries(arm_fds[arm], expected_entries, f"reproducibility evidence arm {arm}")
        receipt = {
            "schema_version": REPRO_BUILD_INPUTS_VERSION,
            "status": "prepared",
            "qualification_status": "not-run",
            "capture_scope": CAPTURE_SCOPE,
            "baseline_id": source_inputs.RECONSTRUCTED_RC2_BASELINE_ID,
            "source_inputs": source_projection,
            "reproducibility_contract": contract,
            "builds": {
                arm: {
                    "reconstruction_id": arm,
                    "evidence_build_id": BUILD_IDS[arm],
                    **{field: descriptors[arm][field].as_json() for field, _leaf, _member, _maximum in BUILD_LEAVES},
                }
                for arm in RECONSTRUCTION_IDS
            },
            "equality": {
                field: {
                    "a": descriptors["a"][field].as_json(),
                    "b": descriptors["b"][field].as_json(),
                    "sha256": descriptors["a"][field].sha256,
                }
                for field in ("binary", "bundle")
            },
            "binding_status": dict(BINDING_STATUS),
            "not_established": dict(NOT_ESTABLISHED),
        }
        _common(
            lambda: common.write_create_only_json(
                root_fd,
                REPRO_BUILD_INPUTS_NAME,
                receipt,
                "reproducibility inputs receipt",
            )
        )
        replayed = _verify_reconstructed_repro_build_inputs_fd(
            root_fd,
            source_root_fd,
            expected_source_archive_sha256=expected_source_sha,
            expected_build_image_id=expected_image,
        )
        if replayed != receipt:
            _fail("prepublication-replay-drift", "held reproducibility inputs replay differs from draft receipt")
        return receipt
    finally:
        for arm_fd in arm_fds.values():
            os.close(arm_fd)
        if repro_fd is not None:
            os.close(repro_fd)
        if root_fd is not None:
            os.close(root_fd)
        os.close(source_root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--source-input-root", type=Path, required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--expected-build-image-id", required=True)
    parser.add_argument("--repro-build-a", type=Path, required=True)
    parser.add_argument("--repro-build-b", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = prepare_reconstructed_repro_build_inputs(
            args.evidence_root,
            source_input_root=args.source_input_root,
            expected_source_archive_sha256=args.expected_source_archive_sha256,
            expected_build_image_id=args.expected_build_image_id,
            repro_build_a=args.repro_build_a,
            repro_build_b=args.repro_build_b,
        )
    except ReproBuildInputsError as error:
        print(f"reconstructed reproducibility inputs: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
