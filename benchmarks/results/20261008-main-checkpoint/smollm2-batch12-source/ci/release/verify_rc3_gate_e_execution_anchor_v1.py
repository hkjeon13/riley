#!/usr/bin/env python3
"""Fail-closed verifier for the external RC3 Gate E execution anchor.

The repository checkout is development input, not authority to start a GPU
capture.  Before a future authenticated Gate E producer can run, an operator
must install its bootstrap and private raw core under one root-owned,
non-group/world-writable directory outside that checkout.  This module only
verifies that installation contract.  It does not execute either file, open a
GPU lock, create evidence, or make a Gate E/qualification decision.

The public CLI deliberately has no path override.  Test code may call the
private verifier with a temporary path and an injected owner UID; production
verification always uses the fixed root-owned paths below.

Its ``checked`` output is an installation preflight from a mutable checkout,
not an approval, launch capability, receipt input, or qualification input.  A
future root-installed bootstrap must independently verify its own bytes, host
mount namespace, and ACL policy before it can hold GPU authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NoReturn, Sequence


ANCHOR_SCHEMA_VERSION = "riley.rc3-gate-e-execution-anchor.v1"
PROBE_SCHEMA_VERSION = "riley.rc3-gate-e-execution-anchor-preflight.v1"
ANCHOR_SCOPE = "execution-anchor-installation-preflight-only"
ANCHOR_AUTHORITY = "execution-anchor-installation-preflight-only"
ANCHOR_STATUS = "checked"

ANCHOR_ROOT = Path("/opt/riley/rc3-gate-e-v1")
LOCK_DIRECTORY = Path("/var/lib/riley/rc3-gate-e/lock")
PYTHON_PATH = "/usr/bin/python3.10"
PYTHON_SHA256 = "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
MANIFEST_NAME = "execution-anchor.json"
BOOTSTRAP_NAME = "run_remote_rc3_gate_e_session_v3.py"
CORE_NAME = "rc3_gate_e_private_raw_core_v1.py"

MAX_MANIFEST_BYTES = 64 * 1024
MAX_CODE_BYTES = 2 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ExecutionAnchorError(ValueError):
    """The external execution anchor cannot safely be trusted."""


def _fail(code: str, message: str) -> NoReturn:
    error = ExecutionAnchorError(f"{code}: {message}")
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail("noncanonical-json", f"cannot encode anchor manifest: {error}")


def _sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb", buffering=0) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        _fail("unsafe-reviewed-python", f"cannot hash reviewed Python: {error}")
    return digest.hexdigest()


def _require_isolated_reviewed_python() -> None:
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.ignore_environment):
        _fail("unsafe-reviewed-python", "must run under /usr/bin/python3.10 -I -S -E")
    try:
        same_executable = os.path.samefile("/proc/self/exe", PYTHON_PATH)
        metadata = os.stat(PYTHON_PATH, follow_symlinks=False)
    except OSError as error:
        _fail("unsafe-reviewed-python", f"cannot authenticate reviewed Python: {error}")
    if (
        not same_executable
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        or _sha256_path(PYTHON_PATH) != PYTHON_SHA256
    ):
        _fail("unsafe-reviewed-python", "reviewed Python executable differs from the pinned contract")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("noncanonical-json", "anchor manifest has duplicate object keys")
        result[key] = value
    return result


def _safe_open_flags() -> tuple[int, int, int, int]:
    values: list[int] = []
    for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC", "O_NONBLOCK"):
        value = getattr(os, name, None)
        if type(value) is not int or value <= 0:
            _fail("missing-safe-open-flag", f"platform lacks required os.{name}")
        values.append(value)
    return values[0], values[1], values[2], values[3]


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _require_directory(
    metadata: os.stat_result,
    label: str,
    *,
    owner_uid: int,
    exact_mode: int | None = None,
) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("unsafe-anchor-directory", f"{label} must be a directory")
    if metadata.st_uid != owner_uid:
        _fail("unsafe-anchor-owner", f"{label} must be owned by UID {owner_uid}")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        _fail("unsafe-anchor-mode", f"{label} must not be group/world writable")
    if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
        _fail("unsafe-anchor-mode", f"{label} must not carry special mode bits")
    if exact_mode is not None and stat.S_IMODE(metadata.st_mode) != exact_mode:
        _fail("unsafe-anchor-mode", f"{label} must have mode {exact_mode:04o}")


def _require_regular(
    metadata: os.stat_result,
    label: str,
    *,
    owner_uid: int,
    executable: bool,
    max_bytes: int,
) -> None:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        _fail("unsafe-anchor-file", f"{label} must be a single-link regular file")
    if metadata.st_uid != owner_uid:
        _fail("unsafe-anchor-owner", f"{label} must be owned by UID {owner_uid}")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        _fail("unsafe-anchor-mode", f"{label} must not be group/world writable")
    if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
        _fail("unsafe-anchor-mode", f"{label} must not carry special mode bits")
    if executable and not metadata.st_mode & stat.S_IXUSR:
        _fail("unsafe-anchor-mode", f"{label} must be owner executable")
    if metadata.st_size <= 0 or metadata.st_size > max_bytes:
        _fail("unsafe-anchor-size", f"{label} has an unsafe byte length")


def _absolute_components(path: Path, label: str) -> tuple[str, ...]:
    raw = os.fspath(path)
    if (
        type(raw) is not str
        or not raw
        or "\x00" in raw
        or not os.path.isabs(raw)
        or raw.startswith("//")
        or os.path.normpath(raw) != raw
        or raw == os.path.sep
    ):
        _fail("invalid-absolute-path", f"{label} must be a normalized non-root absolute path")
    components = tuple(part for part in raw.split(os.path.sep) if part)
    if not components or any(part in {".", ".."} for part in components):
        _fail("invalid-absolute-path", f"{label} contains unsafe path components")
    return components


def _prefix_components(path: Path, label: str) -> tuple[str, ...]:
    raw = os.fspath(path)
    if (
        type(raw) is not str
        or not raw
        or "\x00" in raw
        or not os.path.isabs(raw)
        or raw.startswith("//")
        or os.path.normpath(raw) != raw
    ):
        _fail("invalid-absolute-path", f"{label} must be a normalized absolute path")
    return tuple(part for part in raw.split(os.path.sep) if part)


def _open_absolute_root_owned_directory(
    path: Path,
    label: str,
    *,
    owner_uid: int,
    final_mode: int | None = None,
    trusted_prefix: Path = Path("/"),
) -> int:
    nofollow, directory, cloexec, _nonblock = _safe_open_flags()
    flags = os.O_RDONLY | nofollow | directory | cloexec
    components = _absolute_components(path, label)
    prefix = _prefix_components(trusted_prefix, f"{label} trusted prefix")
    if components[: len(prefix)] != prefix:
        _fail("invalid-absolute-path", f"{label} is outside its trusted prefix")
    try:
        current_fd = os.open(os.path.sep, flags)
    except OSError as error:
        _fail("unsafe-anchor-directory", f"cannot open filesystem root for {label}: {error}")
    try:
        if not prefix:
            _require_directory(os.fstat(current_fd), "filesystem root", owner_uid=owner_uid)
        for index, component in enumerate(components):
            current_label = f"{label} component {component!r}"
            try:
                before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            except OSError as error:
                _fail("unsafe-anchor-directory", f"cannot inspect {current_label}: {error}")
            if index >= len(prefix):
                _require_directory(
                    before,
                    current_label,
                    owner_uid=owner_uid,
                    exact_mode=final_mode if index == len(components) - 1 else None,
                )
            elif not stat.S_ISDIR(before.st_mode):
                _fail("unsafe-anchor-directory", f"{current_label} must be a directory")
            try:
                child_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as error:
                _fail("unsafe-anchor-directory", f"cannot open {current_label}: {error}")
            try:
                opened = os.fstat(child_fd)
                if index >= len(prefix):
                    _require_directory(
                        opened,
                        current_label,
                        owner_uid=owner_uid,
                        exact_mode=final_mode if index == len(components) - 1 else None,
                    )
                elif not stat.S_ISDIR(opened.st_mode):
                    _fail("unsafe-anchor-directory", f"{current_label} must be a directory")
                if _stable_identity(opened) != _stable_identity(before):
                    _fail("raced-anchor-directory", f"{current_label} changed while opening")
            except BaseException:
                os.close(child_fd)
                raise
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


@dataclass(frozen=True)
class AnchorFile:
    filename: str
    sha256: str
    byte_length: int
    device: int
    inode: int

    def as_json(self) -> dict[str, int | str]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "device": self.device,
            "inode": self.inode,
        }


def _read_hashed_regular(
    directory_fd: int,
    name: str,
    label: str,
    *,
    owner_uid: int,
    executable: bool,
    max_bytes: int,
    collect: bool,
) -> tuple[AnchorFile, bytes]:
    if type(name) is not str or not name or os.path.basename(name) != name:
        _fail("invalid-anchor-filename", f"{label} must be a direct filename")
    nofollow, _directory, cloexec, nonblock = _safe_open_flags()
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        _fail("missing-anchor-file", f"cannot inspect {label}: {error}")
    _require_regular(
        before,
        label,
        owner_uid=owner_uid,
        executable=executable,
        max_bytes=max_bytes,
    )
    try:
        descriptor = os.open(name, os.O_RDONLY | nofollow | cloexec | nonblock, dir_fd=directory_fd)
    except OSError as error:
        _fail("unsafe-anchor-file", f"cannot open {label} without following links: {error}")
    try:
        opened = os.fstat(descriptor)
        _require_regular(
            opened,
            label,
            owner_uid=owner_uid,
            executable=executable,
            max_bytes=max_bytes,
        )
        if _stable_identity(opened) != _stable_identity(before):
            _fail("raced-anchor-file", f"{label} changed while opening")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        seen = 0
        while True:
            try:
                block = os.read(descriptor, 1024 * 1024)
            except OSError as error:
                _fail("unreadable-anchor-file", f"cannot read {label}: {error}")
            if not block:
                break
            seen += len(block)
            if seen > max_bytes:
                _fail("unsafe-anchor-size", f"{label} exceeded its byte limit")
            digest.update(block)
            if collect:
                chunks.append(block)
        after = os.fstat(descriptor)
        if _stable_identity(after) != _stable_identity(opened) or seen != opened.st_size:
            _fail("raced-anchor-file", f"{label} changed while hashing")
    finally:
        os.close(descriptor)
    try:
        named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        _fail("raced-anchor-file", f"cannot re-inspect {label}: {error}")
    if _stable_identity(named_after) != _stable_identity(before):
        _fail("raced-anchor-file", f"{label} changed while hashing")
    return (
        AnchorFile(
            filename=name,
            sha256=digest.hexdigest(),
            byte_length=seen,
            device=before.st_dev,
            inode=before.st_ino,
        ),
        b"".join(chunks) if collect else b"",
    )


def _parse_manifest(raw: bytes) -> dict[str, object]:
    if not raw.endswith(b"\n"):
        _fail("noncanonical-json", "execution anchor manifest must end in one newline")
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("invalid-anchor-manifest", f"cannot parse execution anchor manifest: {error}")
    if type(value) is not dict or _canonical_json_bytes(value) + b"\n" != raw:
        _fail("noncanonical-json", "execution anchor manifest is not canonical JSON")
    return value


def _manifest_file_spec(
    manifest: Mapping[str, object],
    field: str,
    expected_name: str,
) -> tuple[str, int]:
    value = manifest.get(field)
    if type(value) is not dict or set(value) != {"filename", "sha256", "byte_length"}:
        _fail("invalid-anchor-manifest", f"{field} must have the fixed file descriptor shape")
    filename = value["filename"]
    digest = value["sha256"]
    byte_length = value["byte_length"]
    if filename != expected_name:
        _fail("invalid-anchor-manifest", f"{field}.filename must be {expected_name!r}")
    if type(digest) is not str or SHA256_RE.fullmatch(digest) is None or digest == "0" * 64:
        _fail("invalid-anchor-manifest", f"{field}.sha256 must be a non-zero lowercase SHA-256")
    if type(byte_length) is not int or byte_length <= 0 or byte_length > MAX_CODE_BYTES:
        _fail("invalid-anchor-manifest", f"{field}.byte_length is unsafe")
    return digest, byte_length


def _verify_anchor(
    anchor_root: Path,
    lock_directory: Path,
    *,
    owner_uid: int,
    trusted_prefix: Path = Path("/"),
) -> dict[str, object]:
    """Verify an injected anchor tree; private test seam only.

    Production callers must use :func:`verify_fixed_execution_anchor`, which
    fixes all paths and the required root owner UID to zero.
    """

    root_fd = _open_absolute_root_owned_directory(
        anchor_root,
        "execution anchor root",
        owner_uid=owner_uid,
        final_mode=0o755,
        trusted_prefix=trusted_prefix,
    )
    try:
        manifest_file, manifest_raw = _read_hashed_regular(
            root_fd,
            MANIFEST_NAME,
            "execution anchor manifest",
            owner_uid=owner_uid,
            executable=False,
            max_bytes=MAX_MANIFEST_BYTES,
            collect=True,
        )
        manifest = _parse_manifest(manifest_raw)
        if set(manifest) != {"schema_version", "bootstrap", "core", "lock_directory"}:
            _fail("invalid-anchor-manifest", "execution anchor manifest has unexpected fields")
        if manifest["schema_version"] != ANCHOR_SCHEMA_VERSION:
            _fail("invalid-anchor-manifest", "execution anchor manifest schema_version differs")
        if manifest["lock_directory"] != os.fspath(lock_directory):
            _fail("invalid-anchor-manifest", "execution anchor manifest lock_directory differs")
        expected_bootstrap_sha, expected_bootstrap_length = _manifest_file_spec(
            manifest,
            "bootstrap",
            BOOTSTRAP_NAME,
        )
        expected_core_sha, expected_core_length = _manifest_file_spec(
            manifest,
            "core",
            CORE_NAME,
        )
        bootstrap, _unused = _read_hashed_regular(
            root_fd,
            BOOTSTRAP_NAME,
            "execution anchor bootstrap",
            owner_uid=owner_uid,
            executable=True,
            max_bytes=MAX_CODE_BYTES,
            collect=False,
        )
        core, _unused = _read_hashed_regular(
            root_fd,
            CORE_NAME,
            "execution anchor private core",
            owner_uid=owner_uid,
            executable=False,
            max_bytes=MAX_CODE_BYTES,
            collect=False,
        )
    finally:
        os.close(root_fd)
    if (
        bootstrap.sha256 != expected_bootstrap_sha
        or bootstrap.byte_length != expected_bootstrap_length
    ):
        _fail("anchor-bootstrap-digest-mismatch", "bootstrap bytes differ from the root-owned manifest")
    if core.sha256 != expected_core_sha or core.byte_length != expected_core_length:
        _fail("anchor-core-digest-mismatch", "private core bytes differ from the root-owned manifest")
    lock_fd = _open_absolute_root_owned_directory(
        lock_directory,
        "execution anchor lock directory",
        owner_uid=owner_uid,
        final_mode=0o700,
        trusted_prefix=trusted_prefix,
    )
    try:
        lock_metadata = os.fstat(lock_fd)
        lock_identity = {"device": lock_metadata.st_dev, "inode": lock_metadata.st_ino}
    finally:
        os.close(lock_fd)
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "anchor_manifest_schema_version": ANCHOR_SCHEMA_VERSION,
        "scope": ANCHOR_SCOPE,
        "authority": ANCHOR_AUTHORITY,
        "status": ANCHOR_STATUS,
        "anchor_root": os.fspath(anchor_root),
        "manifest": manifest_file.as_json(),
        "bootstrap": bootstrap.as_json(),
        "core": core.as_json(),
        "lock_directory": {
            "path": os.fspath(lock_directory),
            **lock_identity,
        },
        "checks": [
            "fixed-root-owned-non-writable-ancestor-chain",
            "canonical-root-owned-manifest",
            "bootstrap-and-core-fd-hashed-against-manifest",
            "fixed-private-root-owned-lock-directory",
            "isolated-reviewed-interpreter-required-for-public-probe",
            "verification-is-no-action-only",
        ],
        "not_established": {
            "bootstrap_execution": "not-established",
            "private_core_execution": "not-established",
            "verifier_source_integrity": "not-established",
            "host_mount_namespace_identity": "not-established",
            "acl_write_prohibition": "not-established",
            "gpu_lock_acquired": "not-established",
            "gpu_query": "not-established",
            "docker_execution": "not-established",
            "evidence_created": "not-established",
            "actual_gate_e_capture": "not-established",
            "semantic_receipt": "not-established",
            "qualification": "not-established",
        },
    }


def verify_fixed_execution_anchor() -> dict[str, object]:
    """Check the only fixed anchor location; do not add path inputs.

    This function is an installation preflight, not execution authority.  A
    future root-installed bootstrap must repeat its own host-context checks.
    """

    return _verify_anchor(ANCHOR_ROOT, LOCK_DIRECTORY, owner_uid=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="verify the fixed root-owned RC3 Gate E execution anchor"
    )
    parser.add_argument(
        "--anchor-contract-probe",
        action="store_true",
        help="verify only the fixed immutable external anchor; no action is run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.anchor_contract_probe:
        _parser().error("--anchor-contract-probe is required")
    try:
        _require_isolated_reviewed_python()
        result = verify_fixed_execution_anchor()
    except ExecutionAnchorError as error:
        print(f"RC3 Gate E execution anchor: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
