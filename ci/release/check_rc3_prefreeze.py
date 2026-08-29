#!/usr/bin/env python3
"""Fail closed on a source checkout before an RC3 candidate is frozen.

This checker deliberately has no output path and never creates a candidate,
archive, image, ELF, raw receipt, or freeze hash.  It only returns a
point-in-time, source-pre-freeze report after checking a clean Git checkout
and reading the source inputs through held no-follow directory descriptors.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

import provenance_v2_common as common


sys.dont_write_bytecode = True

PREFREEZE_REPORT_VERSION = "riley.rc3-prefreeze-check.v1"
MAX_SOURCE_INPUT_BYTES = 8 * 1024 * 1024
MAX_WORKSPACE_MEMBERS = 128
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
CANDIDATE_ID_RE = re.compile(
    r"^riley-((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*))-rc([1-9][0-9]*)$"
)
TABLE_HEADER_RE = re.compile(r"(?m)^[ \t]*\[([A-Za-z0-9_.-]+)\][ \t]*(?:#.*)?$")

# These are duplicated narrowly rather than importing ``release_common``:
# the remote pre-freeze host is Python 3.10, while that build-oriented module
# imports Python 3.11's ``tomllib``.  Keeping this source-only checker
# stdlib-3.10-compatible also avoids importing a path-reading helper before
# the held-FD/no-follow boundary is established.
MIT_LICENSE_EXPRESSION = "MIT"
SERVER_DEFAULTS_SOURCE_PATH = "crates/riley-server/src/main.rs"
# Reviewed at source commit 21f445f4870a140346509144c36c7294f2f677f3.
# Relative to the prior 1195cf20e source pin, ordinary serve defaults remain
# unchanged; the C02 runtime-config, audit, shutdown, and native-fallback
# paths are opt-in and fail closed.  Do not derive or relax this value at
# runtime: a future source change requires another release-contract review.
SERVER_DEFAULTS_SOURCE_SHA256 = (
    "47990249835eed190ee73521ede239841eae0eb73f20e71577258790f1734e4b"
)


class Rc3PrefreezeError(ValueError):
    """The submitted source snapshot cannot safely be checked pre-freeze."""


def _fail(code: str, message: str) -> NoReturn:
    error = Rc3PrefreezeError(f"{code}: {message}")
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Any) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-source-path"), str(error))


def _repository_path(value: Path) -> Path:
    """Return an absolute, lexical path without resolving any symlinks."""

    try:
        raw = os.fspath(value)
    except TypeError as error:
        _fail("invalid-repository-root", f"--repository-root is not a path: {error}")
    if not raw or "\x00" in raw:
        _fail("invalid-repository-root", "--repository-root must be a non-empty path")
    if not os.path.isabs(raw):
        raw = os.path.join(os.getcwd(), raw)
    normalized = os.path.normpath(raw)
    if raw != normalized or raw.startswith("//"):
        _fail(
            "invalid-repository-root",
            "--repository-root must be an absolute normalized lexical path",
        )
    return Path(raw)


def _revision(value: str) -> str:
    if (
        type(value) is not str
        or REVISION_RE.fullmatch(value) is None
        or value == "0" * 40
    ):
        _fail(
            "invalid-expected-revision",
            "--expected-revision must be a full lowercase 40-character Git SHA, not an alias",
        )
    return value


def _candidate_id(value: str) -> tuple[str, str]:
    if type(value) is not str:
        _fail("invalid-candidate-id", "--candidate-id must be text")
    match = CANDIDATE_ID_RE.fullmatch(value)
    if match is None:
        _fail("invalid-candidate-id", "--candidate-id must be canonical riley-X.Y.Z-rcN")
    return value, match.group(1)


def _git_environment() -> dict[str, str]:
    """Use Git only as a read-only source identity/status oracle.

    An inherited ``GIT_*`` override could make an apparently local query point
    at a different index or work tree.  Drop those overrides and disable
    optional Git locks so the checker remains source-only.
    """

    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return environment


def _run_git(
    repository_root: Path,
    root_fd: int,
    arguments: Sequence[str],
    label: str,
) -> bytes:
    """Run a read-only Git query rooted at the caller-held checkout FD."""

    pinned_root = f"/proc/self/fd/{root_fd}"
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-c",
                "maintenance.auto=false",
                "-c",
                "core.fileMode=true",
                "-c",
                "core.trustctime=true",
                "-c",
                "core.checkStat=default",
                "-c",
                "core.ignoreStat=false",
                "-c",
                "core.ignorecase=false",
                "-c",
                "core.symlinks=true",
                "-C",
                pinned_root,
                *arguments,
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_git_environment(),
            pass_fds=(root_fd,),
        )
    except (OSError, ValueError) as error:
        _fail("git-unavailable", f"cannot run git for {label}: {error}")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        _fail("git-command-failed", f"git {label} failed: {detail or completed.returncode}")
    return completed.stdout


def _git_single_line(
    repository_root: Path,
    root_fd: int,
    arguments: Sequence[str],
    label: str,
) -> str:
    raw = _run_git(repository_root, root_fd, arguments, label)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail("invalid-git-output", f"git {label} was not UTF-8: {error}")
    if not text.endswith("\n") or "\n" in text[:-1] or not text[:-1]:
        _fail("invalid-git-output", f"git {label} did not return one non-empty line")
    return text[:-1]


def _check_index_visibility(repository_root: Path, root_fd: int) -> None:
    """Reject index flags that make a tracked source leaf invisible to status.

    ``assume-unchanged`` and ``skip-worktree`` can make ``git status`` report
    a clean checkout after bytes have changed.  A source-revision claim may
    not rely on either optimization, so every NUL-delimited tracked entry must
    have Git's ordinary uppercase ``H`` visibility tag.
    """

    raw = _run_git(
        repository_root,
        root_fd,
        ("ls-files", "-v", "-z"),
        "index visibility",
    )
    if not raw or not raw.endswith(b"\0"):
        _fail("invalid-git-output", "git index visibility output is malformed")
    for entry in raw[:-1].split(b"\0"):
        if len(entry) < 3 or entry[:2] != b"H ":
            _fail(
                "unsafe-index-flags",
                "all tracked source paths must be ordinary visible index entries",
            )


def _check_git_snapshot(repository_root: Path, root_fd: int, expected_revision: str) -> None:
    top_level = _git_single_line(
        repository_root,
        root_fd,
        ("rev-parse", "--show-toplevel"),
        "repository root",
    )
    if top_level != os.fspath(repository_root):
        _fail(
            "repository-root-mismatch",
            "--repository-root must be the Git top-level with the same lexical path",
        )
    actual_revision = _git_single_line(
        repository_root,
        root_fd,
        ("rev-parse", "--verify", "HEAD^{commit}"),
        "HEAD revision",
    )
    if REVISION_RE.fullmatch(actual_revision) is None:
        _fail("invalid-git-output", "git HEAD did not resolve to a full lowercase SHA")
    if actual_revision != expected_revision:
        _fail(
            "head-mismatch",
            f"current HEAD {actual_revision} differs from --expected-revision {expected_revision}",
        )
    dirty = _run_git(
        repository_root,
        root_fd,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=no",
            "--ignore-submodules=none",
        ),
        "checkout status",
    )
    if dirty:
        _fail(
            "checkout-not-clean",
            "tracked or non-ignored untracked source paths are present before freeze",
        )
    _check_index_visibility(repository_root, root_fd)


def _read_source_leaf(
    root_fd: int,
    relative_path: str,
    label: str,
) -> tuple[bytes, common.EvidenceDescriptor]:
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            relative_path,
            label,
            maximum_bytes=MAX_SOURCE_INPUT_BYTES,
        )
    )
    descriptor = _common(
        lambda: common.descriptor_for_bytes(relative_path, raw, label)
    )
    return raw, descriptor


def _utf8_text(raw: bytes, label: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail("invalid-toml", f"{label} is not valid UTF-8 TOML: {error}")


def _reject_ambiguous_toml_syntax(text: str, label: str) -> None:
    """Reject TOML forms this narrow Python-3.10 metadata parser cannot own.

    In particular, a regex-only table search could mistake a table-shaped line
    inside a multiline string for actual metadata.  Cargo manifests in this
    release contract do not need multiline strings.  We reject them and also
    require array/inline-table delimiters to be balanced before a recognized
    table header, rather than attempting a partial permissive TOML parser.
    """

    if '\"\"\"' in text or "'''" in text:
        _fail(
            "unsupported-toml-syntax",
            f"{label} contains multiline TOML strings, which are not accepted pre-freeze",
        )
    square_depth = 0
    curly_depth = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if TABLE_HEADER_RE.fullmatch(line) is not None:
            if square_depth or curly_depth:
                _fail(
                    "unsupported-toml-syntax",
                    f"{label} has a table header inside an unfinished value at line {line_number}",
                )
            continue
        offset = 0
        while offset < len(line):
            character = line[offset]
            if character == "#":
                break
            if character == '"':
                offset += 1
                while offset < len(line):
                    if line[offset] == "\\":
                        offset += 2
                        continue
                    if line[offset] == '"':
                        offset += 1
                        break
                    offset += 1
                else:
                    _fail(
                        "unsupported-toml-syntax",
                        f"{label} has an unterminated basic string at line {line_number}",
                    )
                continue
            if character == "'":
                end = line.find("'", offset + 1)
                if end < 0:
                    _fail(
                        "unsupported-toml-syntax",
                        f"{label} has an unterminated literal string at line {line_number}",
                    )
                offset = end + 1
                continue
            if character == "[":
                square_depth += 1
            elif character == "]":
                square_depth -= 1
                if square_depth < 0:
                    _fail(
                        "unsupported-toml-syntax",
                        f"{label} has an unmatched closing array delimiter at line {line_number}",
                    )
            elif character == "{":
                curly_depth += 1
            elif character == "}":
                curly_depth -= 1
                if curly_depth < 0:
                    _fail(
                        "unsupported-toml-syntax",
                        f"{label} has an unmatched closing inline-table delimiter at line {line_number}",
                    )
            offset += 1
    if square_depth or curly_depth:
        _fail(
            "unsupported-toml-syntax",
            f"{label} has an unfinished array or inline table",
        )


def _table_body(text: str, table_name: str, label: str) -> str:
    headers = list(TABLE_HEADER_RE.finditer(text))
    matches = [header for header in headers if header.group(1) == table_name]
    if len(matches) != 1:
        _fail("invalid-release-metadata", f"{label} must contain exactly one [{table_name}] table")
    header = matches[0]
    next_headers = [candidate for candidate in headers if candidate.start() > header.start()]
    end = next_headers[0].start() if next_headers else len(text)
    return text[header.end() : end]


def _quoted_assignment(section: str, key: str, label: str) -> str:
    pattern = re.compile(
        rf'(?m)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*"([^"\\\r\n]*)"[ \t]*(?:#.*)?$'
    )
    values = pattern.findall(section)
    if len(values) != 1:
        _fail("invalid-release-metadata", f"{label} must be assigned once as a basic TOML string")
    return values[0]


def _has_assignment(section: str, key: str) -> bool:
    return (
        re.search(
            rf'(?m)^[ \t]*(?:{re.escape(key)}|"{re.escape(key)}")[ \t]*(?:=|\.)',
            section,
        )
        is not None
    )


def _has_bare_assignment(section: str, key: str) -> bool:
    return (
        re.search(
            rf'(?m)^[ \t]*(?:{re.escape(key)}|"{re.escape(key)}")[ \t]*=',
            section,
        )
        is not None
    )


def _reject_quoted_keys(section: str, label: str) -> None:
    """Reject escaped/quoted TOML keys from the narrow metadata grammar.

    TOML permits escaped quoted keys such as ``"license\\u002dfile"``.
    Treating only its lexical spelling as a distinct key would let it evade a
    ``license-file`` prohibition.  These release-contract tables have no need
    for quoted keys, so rejecting them is safer than partially decoding TOML.
    """

    if re.search(
        r'''(?m)^[ \t]*(?:"(?:[^"\\]|\\.)*"|'[^'\r\n]*')[ \t]*(?:=|\.)''',
        section,
    ) is not None:
        _fail(
            "invalid-release-metadata",
            f"{label} must not use quoted TOML keys",
        )


def _skip_toml_trivia(value: str, offset: int) -> int:
    while offset < len(value):
        if value[offset] in " \t\r\n":
            offset += 1
            continue
        if value[offset] == "#":
            newline = value.find("\n", offset)
            return len(value) if newline < 0 else newline + 1
        return offset
    return offset


def _parse_member_array(body: str) -> list[str]:
    """Parse the narrow basic-string TOML array accepted for workspace.members."""

    members: list[str] = []
    offset = 0
    while True:
        offset = _skip_toml_trivia(body, offset)
        if offset == len(body):
            return members
        if body[offset] != '"':
            _fail(
                "invalid-release-metadata",
                "workspace.members must contain only basic string literals",
            )
        end = body.find('"', offset + 1)
        if end < 0 or "\\" in body[offset + 1 : end]:
            _fail(
                "invalid-release-metadata",
                "workspace.members must use unescaped basic string literals",
            )
        member = body[offset + 1 : end]
        if not member:
            _fail("invalid-release-metadata", "workspace.members must not include an empty path")
        members.append(member)
        offset = _skip_toml_trivia(body, end + 1)
        if offset == len(body):
            return members
        if body[offset] != ",":
            _fail(
                "invalid-release-metadata",
                "workspace.members values must be comma-separated",
            )
        offset += 1


def _workspace_members(section: str) -> list[str]:
    matches = list(
        re.finditer(
            r"(?ms)^[ \t]*members[ \t]*=[ \t]*\[(.*?)\][ \t]*(?:#.*)?$",
            section,
        )
    )
    if len(matches) != 1:
        _fail("invalid-release-metadata", "workspace.members must be one explicit array")
    members = _parse_member_array(matches[0].group(1))
    if len(members) > MAX_WORKSPACE_MEMBERS:
        _fail("invalid-release-metadata", "workspace.members exceeds its supported bound")
    return members


def _validate_workspace_metadata(root_fd: int) -> tuple[str, list[common.EvidenceDescriptor]]:
    raw, root_descriptor = _read_source_leaf(root_fd, "Cargo.toml", "workspace manifest")
    text = _utf8_text(raw, "workspace manifest")
    _reject_ambiguous_toml_syntax(text, "workspace manifest")
    workspace = _table_body(text, "workspace", "Cargo.toml")
    package = _table_body(text, "workspace.package", "Cargo.toml")
    _reject_quoted_keys(workspace, "Cargo.toml [workspace]")
    _reject_quoted_keys(package, "Cargo.toml [workspace.package]")
    if _quoted_assignment(package, "license", "workspace.package.license") != MIT_LICENSE_EXPRESSION:
        _fail(
            "invalid-release-metadata",
            'workspace.package.license must exactly equal the reviewed SPDX expression "MIT"',
        )
    if _has_assignment(package, "license-file"):
        _fail(
            "invalid-release-metadata",
            "workspace.package.license-file is forbidden for the release contract",
        )
    version = _quoted_assignment(package, "version", "workspace.package.version")
    if SEMVER_RE.fullmatch(version) is None:
        _fail("invalid-workspace-version", "workspace.package.version is not a supported semantic version")

    members = _workspace_members(workspace)
    member_descriptors = [root_descriptor]
    seen_members: set[str] = set()
    for member in members:
        safe_member = _common(
            lambda item=member: common.validate_relative_path(
                item,
                "workspace member",
            )
        )
        if safe_member in seen_members:
            _fail("invalid-release-metadata", "workspace.members must not repeat a path")
        seen_members.add(safe_member)
        relative_manifest = f"{safe_member}/Cargo.toml"
        member_raw, member_descriptor = _read_source_leaf(
            root_fd,
            relative_manifest,
            f"workspace member manifest {safe_member!r}",
        )
        member_text = _utf8_text(member_raw, relative_manifest)
        _reject_ambiguous_toml_syntax(member_text, relative_manifest)
        member_package = _table_body(
            member_text,
            "package",
            relative_manifest,
        )
        _reject_quoted_keys(member_package, f"{relative_manifest} [package]")
        license_workspace = re.findall(
            r"(?m)^[ \t]*license\.workspace[ \t]*=[ \t]*(true|false)[ \t]*(?:#.*)?$",
            member_package,
        )
        if license_workspace != ["true"] or _has_bare_assignment(member_package, "license"):
            _fail(
                "invalid-release-metadata",
                f"{relative_manifest}: package.license must be license.workspace = true",
            )
        if _has_assignment(member_package, "license-file"):
            _fail(
                "invalid-release-metadata",
                f"{relative_manifest}: package.license-file is forbidden",
            )
        member_descriptors.append(member_descriptor)
    return version, member_descriptors


def _validate_server_defaults(root_fd: int) -> common.EvidenceDescriptor:
    raw, descriptor = _read_source_leaf(
        root_fd,
        SERVER_DEFAULTS_SOURCE_PATH,
        "reviewed server defaults source",
    )
    if descriptor.sha256 != SERVER_DEFAULTS_SOURCE_SHA256:
        _fail(
            "server-defaults-source-mismatch",
            "Rust serve defaults changed without a reviewed release-contract update",
        )
    return descriptor


def _root_still_matches(repository_root: Path, held_root_fd: int) -> None:
    """Reject a path replacement between Git checks and held-FD source reads."""

    reopened_fd = -1
    try:
        reopened_fd = _common(
            lambda: common.open_absolute_directory(repository_root, "repository root")
        )
        held = os.fstat(held_root_fd)
        reopened = os.fstat(reopened_fd)
    except OSError as error:
        _fail("raced-repository-root", f"cannot inspect --repository-root: {error}")
    finally:
        if reopened_fd >= 0:
            os.close(reopened_fd)
    if (held.st_dev, held.st_ino) != (reopened.st_dev, reopened.st_ino):
        _fail("raced-repository-root", "--repository-root changed while it was checked")


def check_prefreeze(
    repository_root: Path,
    expected_revision: str,
    candidate_id: str,
) -> dict[str, Any]:
    """Return only source-pre-freeze facts for one clean, exact source revision."""

    root = _repository_path(repository_root)
    revision = _revision(expected_revision)
    candidate, candidate_version = _candidate_id(candidate_id)
    root_fd = -1
    try:
        root_fd = _common(
            lambda: common.open_absolute_directory(root, "repository root")
        )
        _check_git_snapshot(root, root_fd, revision)
        workspace_version, workspace_manifests = _validate_workspace_metadata(root_fd)
        if workspace_version != candidate_version:
            _fail(
                "candidate-workspace-version-mismatch",
                f"candidate version {candidate_version} differs from workspace version {workspace_version}",
            )
        _cargo_lock_raw, cargo_lock = _read_source_leaf(root_fd, "Cargo.lock", "Cargo.lock")
        _registry_raw, extension_registry = _read_source_leaf(
            root_fd,
            "deploy/extensions/registry.json",
            "extension registry",
        )
        server_defaults = _validate_server_defaults(root_fd)
        _check_git_snapshot(root, root_fd, revision)
        _root_still_matches(root, root_fd)
    finally:
        if root_fd >= 0:
            os.close(root_fd)

    return {
        "schema_version": PREFREEZE_REPORT_VERSION,
        "scope": "source-pre-freeze-only",
        "candidate_status": "not-frozen",
        "qualification_status": "not-run",
        "candidate_id": candidate,
        "source_revision": revision,
        "workspace_version": workspace_version,
        "source_inputs": {
            "workspace_manifests": [
                descriptor.as_json() for descriptor in workspace_manifests
            ],
            "cargo_lock": cargo_lock.as_json(),
            "extension_registry": extension_registry.as_json(),
            "server_defaults_source": server_defaults.as_json(),
        },
        "checks": [
            {"name": "full-source-revision", "satisfied": True},
            {"name": "clean-tracked-and-untracked-checkout", "satisfied": True},
            {"name": "release-metadata-and-workspace-version", "satisfied": True},
            {"name": "no-follow-source-input-hashes", "satisfied": True},
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--candidate-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = check_prefreeze(
            args.repository_root,
            args.expected_revision,
            args.candidate_id,
        )
    except (OSError, Rc3PrefreezeError) as error:
        print(f"RC3 source pre-freeze check failed: {error}", file=sys.stderr)
        return 1
    print(common.canonical_json_bytes(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
