#!/usr/bin/env python3
"""Prepare source-only reviewed inputs for the reconstructed RC2 baseline.

This is deliberately not a reconstructed baseline producer.  It observes one
reviewed annotated RC2 tag from the local Git object store, generates the
canonical archive for its direct target commit, and accepts that archive only
when a caller supplies an independently reviewed SHA-256.  It then writes the
three v2-compatible source leaves and one preparation receipt to a new,
external, private evidence root.

The receipt is not its own archive-review authority: every replay caller must
provide the same independently reviewed SHA-256 again.  The producer validates
the deterministic Git tar grammar while it owns the temporary output; the
receipt intentionally does not carry a self-authored source-date epoch, so a
later A/B builder must derive that fact from the held archive itself.

It does not run a build, construct an image, start a service, inspect a GPU,
or make a rollback/qualification claim.  Those later stages must consume this
receipt through their own reviewed contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import subprocess
import sys

sys.dont_write_bytecode = True

import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence

import check_reconstructed_prior_baseline_v2 as baseline
import provenance_v2_common as common


SOURCE_INPUTS_VERSION = "riley.reconstructed-rc2-source-inputs.v1"
SOURCE_INPUTS_NAME = "reconstructed-rc2-source-inputs.json"
SOURCE_DIRECTORY_NAME = "source"
SOURCE_ARCHIVE_NAME = "riley-0.1.0-rc2.tar"
ARCHIVE_GENERATION = "git-archive-target-commit"
SOURCE_LEAVES = (
    ("tag_object", "git-tag-object.json", "RC2 raw Git tag object"),
    ("tag_target", "git-tag-target.json", "RC2 raw Git tag target"),
    ("archive", SOURCE_ARCHIVE_NAME, "RC2 canonical source archive"),
)

RECONSTRUCTED_RC2_TAG = "riley-0.1.0-rc2"
RECONSTRUCTED_RC2_TAG_REF = f"refs/tags/{RECONSTRUCTED_RC2_TAG}"
RECONSTRUCTED_RC2_TAG_OBJECT = "a3f5203c3a72122e9da818c1e441c2a789f7aa8c"
RECONSTRUCTED_RC2_TARGET = "6093006ec2b01b784b01ba278296b676f2dfd03a"
RECONSTRUCTED_RC2_BASELINE_ID = f"reconstructed-{RECONSTRUCTED_RC2_TAG}"

MAX_SOURCE_MEMBERS = 20_000
MAX_SOURCE_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_SOURCE_ARCHIVE_BYTES = MAX_SOURCE_TOTAL_BYTES + 16 * 1024 * 1024
TAR_BLOCK_BYTES = 512
TAR_RECORD_BYTES = 10_240
REQUIRED_SOURCE_INPUTS = {"Cargo.lock", "Cargo.toml", "ci/release/Dockerfile"}
MAX_RECEIPT_BYTES = common.DEFAULT_MAX_JSON_BYTES
GIT_SHA1_RE = re.compile(r"[0-9a-f]{40}")


class ReconstructedRc2InputsError(common.ProvenanceV2Error):
    """The reviewed RC2 source inputs cannot be prepared or replayed."""


def _fail(code: str, message: str) -> NoReturn:
    error = ReconstructedRc2InputsError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Any) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _baseline(call: Any) -> Any:
    try:
        return call()
    except baseline.BaselineError as error:
        _fail(getattr(error, "reason_code", "invalid-v2-source-binding"), str(error))
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "invalid-v2-source-binding"), str(error))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_environment() -> dict[str, str]:
    """Run only Git builtins without inheriting ambient repository selectors."""

    return {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "TZ": "UTC",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _git_command(repository: Path, *arguments: str) -> list[str]:
    return ["git", "-C", os.fspath(repository), *arguments]


def _run_git(repository: Path, *arguments: str, stdin: Any | None = None) -> bytes:
    try:
        result = subprocess.run(
            _git_command(repository, *arguments),
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=_git_environment(),
        )
    except OSError as error:
        _fail("missing-git", f"cannot execute git for reviewed RC2 inputs: {error}")
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        _fail("git-command-failed", f"git {' '.join(arguments)!r} failed: {detail}")
    return result.stdout


def _one_ascii_line(value: bytes, label: str) -> str:
    try:
        text = value.decode("ascii")
    except UnicodeDecodeError as error:
        _fail("invalid-git-output", f"{label} is not ASCII: {error}")
    if not text.endswith("\n") or text.count("\n") != 1:
        _fail("invalid-git-output", f"{label} must contain one newline-terminated line")
    result = text[:-1]
    if not result:
        _fail("invalid-git-output", f"{label} must not be empty")
    return result


def _git_sha1(value: str, label: str) -> str:
    if GIT_SHA1_RE.fullmatch(value) is None or value == "0" * 40:
        _fail("invalid-git-sha1", f"{label} must be a non-zero lowercase Git SHA-1")
    return value


def _expected_sha256(value: Any) -> str:
    if type(value) is not str or common.SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(
            "invalid-expected-source-archive-sha256",
            "--expected-source-archive-sha256 must be a non-zero lowercase SHA-256",
        )
    return value


def _tag_headers(raw: bytes) -> dict[str, bytes]:
    header_block, separator, _message = raw.partition(b"\n\n")
    if not separator:
        _fail("invalid-annotated-tag", "annotated tag object has no header/message separator")
    fields: dict[str, bytes] = {}
    for line in header_block.splitlines():
        key, space, value = line.partition(b" ")
        if not space or not key or not value:
            _fail("invalid-annotated-tag", "annotated tag object has a malformed header")
        try:
            text_key = key.decode("ascii")
        except UnicodeDecodeError as error:
            _fail("invalid-annotated-tag", f"annotated tag header is not ASCII: {error}")
        if text_key in fields:
            _fail("invalid-annotated-tag", f"annotated tag repeats header {text_key!r}")
        fields[text_key] = value
    return fields


def _observe_reviewed_identity(repository: Path) -> dict[str, Any]:
    tag_object = _git_sha1(
        _one_ascii_line(
            _run_git(repository, "rev-parse", "--verify", RECONSTRUCTED_RC2_TAG_REF),
            "reviewed tag object",
        ),
        "reviewed tag object",
    )
    if tag_object != RECONSTRUCTED_RC2_TAG_OBJECT:
        _fail(
            "reviewed-tag-object-mismatch",
            "local RC2 tag object does not match the reviewed annotated tag object pin",
        )
    object_type = _one_ascii_line(
        _run_git(repository, "cat-file", "-t", tag_object), "reviewed tag object type"
    )
    if object_type != "tag":
        _fail(
            "reviewed-tag-not-annotated",
            "reviewed RC2 ref must resolve to an annotated tag object",
        )
    headers = _tag_headers(_run_git(repository, "cat-file", "-p", tag_object))
    expected_target = RECONSTRUCTED_RC2_TARGET.encode("ascii")
    if headers.get("object") != expected_target or headers.get("type") != b"commit":
        _fail(
            "reviewed-tag-target-mismatch",
            "annotated RC2 tag does not directly name the reviewed target commit",
        )
    if headers.get("tag") != RECONSTRUCTED_RC2_TAG.encode("ascii"):
        _fail("reviewed-tag-name-mismatch", "annotated tag object names a different tag")
    target = _git_sha1(
        _one_ascii_line(
            _run_git(repository, "rev-parse", "--verify", f"{RECONSTRUCTED_RC2_TAG_REF}^{{}}"),
            "reviewed RC2 peeled target",
        ),
        "reviewed RC2 peeled target",
    )
    if target != RECONSTRUCTED_RC2_TARGET:
        _fail(
            "reviewed-tag-target-mismatch",
            "reviewed RC2 tag target does not match the reviewed target commit pin",
        )
    target_type = _one_ascii_line(
        _run_git(repository, "cat-file", "-t", target), "reviewed target object type"
    )
    if target_type != "commit":
        _fail("reviewed-tag-noncommit-target", "reviewed RC2 target must be a commit")
    epoch_text = _one_ascii_line(
        _run_git(repository, "show", "-s", "--format=%ct", target), "reviewed target timestamp"
    )
    if not epoch_text.isascii() or not epoch_text.isdecimal():
        _fail("invalid-source-date-epoch", "reviewed target commit timestamp must be decimal")
    source_date_epoch = int(epoch_text)
    if not 0 <= source_date_epoch <= 0xFFFFFFFF:
        _fail("invalid-source-date-epoch", "reviewed target timestamp is outside the supported range")
    return {
        "tag_ref": RECONSTRUCTED_RC2_TAG_REF,
        "tag_object_sha1": tag_object,
        "target_commit_sha1": target,
        "source_date_epoch": source_date_epoch,
    }


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


def _safe_tar_member_path(name: str) -> PurePosixPath:
    if not name or name.startswith("/") or "\\" in name or "//" in name:
        _fail("source-archive-validation-failed", f"canonical source archive has unsafe member path {name!r}")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts):
        _fail("source-archive-validation-failed", f"canonical source archive has unsafe member path {name!r}")
    return path


def _tar_data_end(member: tarfile.TarInfo) -> int:
    padded_size = ((member.size + TAR_BLOCK_BYTES - 1) // TAR_BLOCK_BYTES) * TAR_BLOCK_BYTES
    return member.offset_data + padded_size


def _validate_tar_end(path: Path, data_end: int) -> None:
    canonical_size = (
        (data_end + 2 * TAR_BLOCK_BYTES + TAR_RECORD_BYTES - 1) // TAR_RECORD_BYTES
    ) * TAR_RECORD_BYTES
    try:
        actual_size = path.stat().st_size
    except OSError as error:
        _fail("source-archive-validation-failed", f"cannot inspect canonical source archive end: {error}")
    if actual_size != canonical_size:
        _fail(
            "source-archive-validation-failed",
            "canonical source archive has non-canonical end-of-archive padding or trailing records",
        )
    try:
        with path.open("rb", buffering=0) as raw:
            raw.seek(data_end)
            padding = raw.read(canonical_size - data_end)
    except OSError as error:
        _fail("source-archive-validation-failed", f"cannot read canonical source archive end: {error}")
    if len(padding) != canonical_size - data_end or any(padding):
        _fail(
            "source-archive-validation-failed",
            "canonical source archive has non-canonical end-of-archive padding or trailing records",
        )


def _validate_canonical_source_archive(
    path: Path,
    target_commit_sha1: str,
    source_date_epoch: int,
) -> None:
    """Require the bounded, deterministic uncompressed Git archive grammar."""

    try:
        metadata = path.lstat()
    except OSError as error:
        _fail("source-archive-validation-failed", f"cannot inspect canonical source archive: {error}")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        _fail("source-archive-validation-failed", "canonical source archive must be a single-link regular file")
    if metadata.st_size < 1 or metadata.st_size > MAX_SOURCE_ARCHIVE_BYTES:
        _fail("source-archive-size", "canonical source archive is empty or exceeds its byte bound")
    names: list[str] = []
    seen_names: set[str] = set()
    previous_archive_name: str | None = None
    last_data_end = 0
    total_size = 0
    try:
        with tarfile.open(path, mode="r:") as archive:
            if archive.pax_headers != {"comment": target_commit_sha1}:
                _fail(
                    "source-archive-validation-failed",
                    "canonical source archive does not embed the exact Git archive revision",
                )
            for member in archive:
                if len(names) >= MAX_SOURCE_MEMBERS:
                    _fail("source-archive-validation-failed", "canonical source archive contains too many members")
                _safe_tar_member_path(member.name)
                if member.name in seen_names:
                    _fail("source-archive-validation-failed", f"canonical source archive repeats {member.name!r}")
                archive_name = member.name + "/" if member.isdir() else member.name
                if previous_archive_name is not None and archive_name < previous_archive_name:
                    _fail(
                        "source-archive-validation-failed",
                        "canonical source archive members are not bytewise sorted",
                    )
                if member.pax_headers != {"comment": target_commit_sha1}:
                    _fail(
                        "source-archive-validation-failed",
                        f"canonical source archive has unreviewed PAX metadata for {member.name!r}",
                    )
                if not (member.isdir() or member.isreg()):
                    _fail(
                        "source-archive-validation-failed",
                        "canonical source archive contains a link or special member",
                    )
                if member.uid != 0 or member.gid != 0 or member.uname != "root" or member.gname != "root":
                    _fail(
                        "source-archive-validation-failed",
                        "canonical source archive ownership differs from Git archive",
                    )
                if member.mtime != source_date_epoch:
                    _fail(
                        "source-archive-validation-failed",
                        "canonical source archive member mtime differs from the reviewed commit timestamp",
                    )
                expected_modes = {0o775} if member.isdir() else {0o664, 0o775}
                if member.mode not in expected_modes:
                    _fail(
                        "source-archive-validation-failed",
                        "canonical source archive member mode differs from Git archive",
                    )
                if member.size > MAX_SOURCE_MEMBER_BYTES:
                    _fail("source-archive-validation-failed", "canonical source archive member exceeds its byte bound")
                total_size += member.size
                if total_size > MAX_SOURCE_TOTAL_BYTES:
                    _fail("source-archive-validation-failed", "canonical source archive exceeds its total byte bound")
                last_data_end = max(last_data_end, _tar_data_end(member))
                names.append(member.name)
                seen_names.add(member.name)
                previous_archive_name = archive_name
    except (OSError, tarfile.TarError) as error:
        _fail("source-archive-validation-failed", f"canonical source archive is not an uncompressed Git tar: {error}")
    _validate_tar_end(path, last_data_end)
    if not names:
        _fail("source-archive-validation-failed", "canonical source archive is empty")
    missing = REQUIRED_SOURCE_INPUTS - {name.removesuffix("/") for name in names}
    if missing:
        _fail(
            "source-archive-validation-failed",
            "canonical source archive is missing release build inputs: " + ", ".join(sorted(missing)),
        )


def _write_bounded_git_archive(
    repository: Path,
    target_commit_sha1: str,
    output: Path,
) -> str:
    """Stream `git archive` into a bounded private temporary file."""

    digest = hashlib.sha256()
    byte_length = 0
    process: Any | None = None
    return_code: int | None = None
    try:
        with output.open("xb", buffering=0) as destination:
            process = subprocess.Popen(
                _git_command(
                    repository,
                    "-c",
                    "tar.umask=0002",
                    "archive",
                    "--format=tar",
                    target_commit_sha1,
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=_git_environment(),
                close_fds=True,
            )
            if process.stdout is None:  # pragma: no cover - subprocess contract
                _fail("source-archive-creation-failed", "git archive did not expose stdout")
            try:
                while True:
                    chunk = process.stdout.read(1024 * 1024)
                    if not chunk:
                        break
                    if byte_length + len(chunk) > MAX_SOURCE_ARCHIVE_BYTES:
                        _fail(
                            "source-archive-size",
                            "canonical source archive exceeds its byte bound",
                        )
                    written = destination.write(chunk)
                    if written != len(chunk):  # pragma: no cover - FileIO contract
                        _fail("source-archive-creation-failed", "cannot fully write canonical source archive")
                    byte_length += len(chunk)
                    digest.update(chunk)
                return_code = process.wait()
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
    except OSError as error:
        _fail("source-archive-creation-failed", f"cannot create canonical RC2 source archive: {error}")
    if return_code != 0:
        _fail("source-archive-creation-failed", "git archive failed to create canonical RC2 source archive")
    if byte_length < 1:
        _fail("source-archive-size", "canonical source archive is empty")
    return digest.hexdigest()


def _create_canonical_source_archive(
    repository: Path,
    target_commit_sha1: str,
    source_date_epoch: int,
    expected_sha256: str,
    output: Path,
) -> None:
    observed_sha256 = _write_bounded_git_archive(repository, target_commit_sha1, output)
    try:
        output.chmod(0o600)
        metadata = output.lstat()
    except OSError as error:
        _fail("source-archive-creation-failed", f"cannot inspect canonical source archive: {error}")
    if not output.is_file() or output.is_symlink() or metadata.st_nlink != 1:
        _fail("source-archive-creation-failed", "canonical source archive must be a single-link regular file")
    if metadata.st_size < 1 or metadata.st_size > MAX_SOURCE_ARCHIVE_BYTES:
        _fail("source-archive-size", "canonical source archive is empty or exceeds its byte bound")
    if observed_sha256 != expected_sha256:
        _fail(
            "source-archive-digest-mismatch",
            "canonical RC2 source archive differs from the externally reviewed SHA-256",
        )
    try:
        with output.open("rb", buffering=0) as source:
            embedded_target = _one_ascii_line(
                _run_git(repository, "get-tar-commit-id", stdin=source),
                "canonical source archive embedded revision",
            )
    except OSError as error:
        _fail("source-archive-validation-failed", f"cannot reopen canonical source archive: {error}")
    if embedded_target != target_commit_sha1:
        _fail(
            "source-archive-target-mismatch",
            "canonical source archive does not embed the reviewed target commit",
        )
    _validate_canonical_source_archive(output, target_commit_sha1, source_date_epoch)


def _require_external_evidence_root(evidence_root: Path, repository: Path) -> None:
    evidence_root = _normalized_absolute_path(evidence_root, "--evidence-root")
    try:
        candidate = evidence_root.parent.resolve(strict=False) / evidence_root.name
        source_root = repository.resolve(strict=True)
    except OSError as error:
        _fail("invalid-evidence-root", f"cannot resolve source/output roots: {error}")
    if candidate == source_root or source_root in candidate.parents:
        _fail("evidence-root-inside-source", "--evidence-root must be outside the source checkout")


def _exact(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else []
        _fail(
            "unknown-or-missing-field",
            f"{label} fields differ; expected={sorted(expected)}, actual={actual}",
        )
    return value


def _parse_receipt(value: Mapping[str, Any]) -> tuple[baseline.SourceBinding, dict[str, Any]]:
    row = _exact(
        dict(value),
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
        "reconstructed RC2 source inputs receipt",
    )
    if (
        row["schema_version"] != SOURCE_INPUTS_VERSION
        or row["status"] != "prepared"
        or row["qualification_status"] != "not-run"
        or row["baseline_id"] != RECONSTRUCTED_RC2_BASELINE_ID
        or row["archive_generation"] != ARCHIVE_GENERATION
    ):
        _fail("invalid-source-inputs-receipt", "source inputs receipt is not the exact v1 contract")
    expected_sha256 = _expected_sha256(row["expected_source_archive_sha256"])
    source = _baseline(lambda: baseline._source(row["source"], "source inputs receipt.source"))
    for field, leaf_name, label in SOURCE_LEAVES:
        descriptor = getattr(source, field)
        expected_path = f"{SOURCE_DIRECTORY_NAME}/{leaf_name}"
        if descriptor.path != expected_path:
            _fail(
                "source-leaf-path-mismatch",
                f"{label} must use the fixed evidence path {expected_path!r}",
            )
    identity = _exact(
        row["git_identity"],
        {"tag_ref", "tag_object_sha1", "target_commit_sha1"},
        "source inputs receipt.git_identity",
    )
    if (
        identity["tag_ref"] != RECONSTRUCTED_RC2_TAG_REF
        or identity["tag_object_sha1"] != RECONSTRUCTED_RC2_TAG_OBJECT
        or identity["target_commit_sha1"] != RECONSTRUCTED_RC2_TARGET
    ):
        _fail("reviewed-identity-mismatch", "source inputs receipt does not retain the reviewed RC2 identity")
    if source.tag_name != RECONSTRUCTED_RC2_TAG or source.archive.sha256 != expected_sha256:
        _fail("source-binding-mismatch", "source receipt does not bind the reviewed RC2 source archive")
    return source, row


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


def _held_source_binding(
    source_fd: int,
    source: baseline.SourceBinding,
) -> baseline.SourceBinding:
    """Rebase the fixed source leaves to the already-held source directory."""

    held: dict[str, common.EvidenceDescriptor] = {}
    for field, leaf_name, label in SOURCE_LEAVES:
        descriptor = getattr(source, field)
        expected_path = f"{SOURCE_DIRECTORY_NAME}/{leaf_name}"
        maximum_bytes = MAX_SOURCE_ARCHIVE_BYTES if field == "archive" else MAX_RECEIPT_BYTES
        held_descriptor = _common(
            lambda descriptor=descriptor, expected_path=expected_path, leaf_name=leaf_name, label=label:
            common.rebase_descriptor_to_held_leaf(
                descriptor,
                expected_root_relative_path=expected_path,
                leaf_name=leaf_name,
                label=label,
            )
        )
        _common(
            lambda held_descriptor=held_descriptor, label=label, maximum_bytes=maximum_bytes: common.verify_private_snapshot_descriptor_file(
                source_fd,
                held_descriptor,
                label,
                maximum_bytes=maximum_bytes,
            )
        )
        held[field] = held_descriptor
    return baseline.SourceBinding(
        tag_name=source.tag_name,
        tag_object=held["tag_object"],
        tag_target=held["tag_target"],
        archive=held["archive"],
    )


def verify_reconstructed_rc2_inputs_fd(
    root_fd: int,
    *,
    expected_source_archive_sha256: str,
) -> dict[str, Any]:
    """Replay one already-held source-inputs root without creating output."""

    expected_sha256 = _expected_sha256(expected_source_archive_sha256)
    _common(lambda: common.require_private_evidence_directory_fd(root_fd, "RC2 source inputs root"))
    _assert_entries(root_fd, {SOURCE_DIRECTORY_NAME, SOURCE_INPUTS_NAME}, "RC2 source inputs root")
    receipt = _common(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            SOURCE_INPUTS_NAME,
            "RC2 source inputs receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    source, row = _parse_receipt(receipt)
    if row["expected_source_archive_sha256"] != expected_sha256:
        _fail(
            "reviewed-source-archive-digest-mismatch",
            "source inputs receipt does not match the caller's reviewed source archive SHA-256",
        )
    source_fd = _common(
        lambda: common.open_private_child_directory(
            root_fd, SOURCE_DIRECTORY_NAME, "RC2 source evidence directory"
        )
    )
    try:
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                source_fd,
                SOURCE_DIRECTORY_NAME,
                "held RC2 source evidence directory",
            )
        )
        _assert_entries(
            source_fd,
            {leaf_name for _field, leaf_name, _label in SOURCE_LEAVES},
            "RC2 source evidence directory",
        )
        held_source = _held_source_binding(source_fd, source)
        identity = _baseline(lambda: baseline._read_tag_identity(source_fd, held_source))
        _assert_entries(
            source_fd,
            {leaf_name for _field, leaf_name, _label in SOURCE_LEAVES},
            "RC2 source evidence directory",
        )
        if _held_source_binding(source_fd, source) != held_source:
            _fail("raced-input", "RC2 source evidence changed during replay")
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                source_fd,
                SOURCE_DIRECTORY_NAME,
                "held RC2 source evidence directory",
            )
        )
    finally:
        os.close(source_fd)
    if (
        identity.tag_name != RECONSTRUCTED_RC2_TAG
        or identity.tag_ref != RECONSTRUCTED_RC2_TAG_REF
        or identity.tag_object_sha1 != RECONSTRUCTED_RC2_TAG_OBJECT
        or identity.target_commit_sha1 != RECONSTRUCTED_RC2_TARGET
    ):
        _fail("reviewed-identity-mismatch", "source evidence leaves do not retain the reviewed RC2 identity")
    _assert_entries(root_fd, {SOURCE_DIRECTORY_NAME, SOURCE_INPUTS_NAME}, "RC2 source inputs root")
    terminal_receipt = _common(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            SOURCE_INPUTS_NAME,
            "RC2 source inputs receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    if terminal_receipt != receipt:
        _fail("raced-input", "RC2 source inputs receipt changed during replay")
    return dict(row)


def verify_reconstructed_rc2_inputs(
    evidence_root: Path,
    *,
    expected_source_archive_sha256: str,
) -> dict[str, Any]:
    evidence_root = _normalized_absolute_path(evidence_root, "RC2 source inputs root")
    root_fd = _common(
        lambda: common.open_private_evidence_directory(evidence_root, "RC2 source inputs root")
    )
    try:
        return verify_reconstructed_rc2_inputs_fd(
            root_fd,
            expected_source_archive_sha256=expected_source_archive_sha256,
        )
    finally:
        os.close(root_fd)


def prepare_reconstructed_rc2_inputs(
    evidence_root: Path,
    *,
    expected_source_archive_sha256: str,
) -> dict[str, Any]:
    """Create one fresh reviewed RC2 source-inputs root and self-replay it."""

    expected_sha256 = _expected_sha256(expected_source_archive_sha256)
    repository = _repository_root()
    evidence_root = _normalized_absolute_path(evidence_root, "--evidence-root")
    _require_external_evidence_root(evidence_root, repository)
    identity = _observe_reviewed_identity(repository)
    with tempfile.TemporaryDirectory(prefix="riley-reconstructed-rc2-source-") as temporary:
        archive = Path(temporary) / SOURCE_ARCHIVE_NAME
        _create_canonical_source_archive(
            repository,
            identity["target_commit_sha1"],
            identity["source_date_epoch"],
            expected_sha256,
            archive,
        )
        root_fd = _common(
            lambda: common.create_private_evidence_directory(evidence_root, "RC2 source inputs root")
        )
        source_fd: int | None = None
        try:
            source_fd = _common(
                lambda: common.create_private_child_directory(
                    root_fd, SOURCE_DIRECTORY_NAME, "RC2 source evidence directory"
                )
            )
            tag_object = _common(
                lambda: common.write_create_only_json(
                    source_fd,
                    "git-tag-object.json",
                    {
                        "schema_version": baseline.GIT_TAG_OBJECT_VERSION,
                        "tag_ref": RECONSTRUCTED_RC2_TAG_REF,
                        "object_type": "tag",
                        "object_sha1": identity["tag_object_sha1"],
                        "target_object_type": "commit",
                        "target_object_sha1": identity["target_commit_sha1"],
                    },
                    "RC2 raw Git tag object observation",
                )
            )
            tag_target = _common(
                lambda: common.write_create_only_json(
                    source_fd,
                    "git-tag-target.json",
                    {
                        "schema_version": baseline.GIT_TAG_TARGET_VERSION,
                        "tag_ref": RECONSTRUCTED_RC2_TAG_REF,
                        "tag_object_sha1": identity["tag_object_sha1"],
                        "target_commit_sha1": identity["target_commit_sha1"],
                    },
                    "RC2 raw Git tag target observation",
                )
            )
            archive_snapshot = _common(
                lambda: common.snapshot_absolute_regular_create_only(
                    archive,
                    source_fd,
                    SOURCE_ARCHIVE_NAME,
                    "RC2 canonical source archive",
                    maximum_bytes=MAX_SOURCE_ARCHIVE_BYTES,
                    minimum_bytes=1,
                )
            )
            source = {
                "tag_name": RECONSTRUCTED_RC2_TAG,
                "tag_object": tag_object.descriptor(
                    f"{SOURCE_DIRECTORY_NAME}/git-tag-object.json", "RC2 raw Git tag object"
                ).as_json(),
                "tag_target": tag_target.descriptor(
                    f"{SOURCE_DIRECTORY_NAME}/git-tag-target.json", "RC2 raw Git tag target"
                ).as_json(),
                "archive": archive_snapshot.descriptor(
                    f"{SOURCE_DIRECTORY_NAME}/{SOURCE_ARCHIVE_NAME}", "RC2 canonical source archive"
                ).as_json(),
            }
            receipt = {
                "schema_version": SOURCE_INPUTS_VERSION,
                "status": "prepared",
                "qualification_status": "not-run",
                "baseline_id": RECONSTRUCTED_RC2_BASELINE_ID,
                "source": source,
                "git_identity": {
                    "tag_ref": identity["tag_ref"],
                    "tag_object_sha1": identity["tag_object_sha1"],
                    "target_commit_sha1": identity["target_commit_sha1"],
                },
                "expected_source_archive_sha256": expected_sha256,
                "archive_generation": ARCHIVE_GENERATION,
            }
            _common(
                lambda: common.write_create_only_json(
                    root_fd,
                    SOURCE_INPUTS_NAME,
                    receipt,
                    "RC2 source inputs receipt",
                )
            )
            replayed = verify_reconstructed_rc2_inputs_fd(
                root_fd,
                expected_source_archive_sha256=expected_sha256,
            )
            if replayed != receipt:
                _fail("prepublication-replay-drift", "held RC2 source-inputs replay differs from draft receipt")
            return receipt
        finally:
            if source_fd is not None:
                os.close(source_fd)
            os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = prepare_reconstructed_rc2_inputs(
            args.evidence_root,
            expected_source_archive_sha256=args.expected_source_archive_sha256,
        )
    except ReconstructedRc2InputsError as error:
        print(f"reconstructed RC2 source inputs: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
