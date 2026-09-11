#!/usr/bin/python3.10
"""Root-installed no-action bootstrap template for the RC3 Gate E v3 core.

This checkout copy is source/audit material only.  The public entrypoint
rejects it before opening an anchor, lock, socket, or child.  A future operator
may install reviewed identical bytes at the fixed root-owned anchor path and
run it only from a clean root-owned service/launcher.  This version performs a
sealed private-core smoke handshake; it has no GPU, Docker, evidence, replay,
receipt, or qualification capability.

The bootstrap deliberately does not import or call the mutable-checkout anchor
verifier.  Its root-owned runtime path repeats the required held-FD, manifest,
ACL, namespace, lock, and process-boundary checks itself.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import stat
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn, Sequence


PYTHON_PATH: Final = "/usr/bin/python3.10"
PYTHON_SHA256: Final = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
ANCHOR_ROOT: Final = Path("/opt/riley/rc3-gate-e-v1")
LOCK_DIRECTORY: Final = Path("/var/lib/riley/rc3-gate-e/lock")
BOOTSTRAP_NAME: Final = "run_remote_rc3_gate_e_session_v3.py"
CORE_NAME: Final = "rc3_gate_e_private_raw_core_v1.py"
MANIFEST_NAME: Final = "execution-anchor.json"
LOCK_NAME: Final = "gate-e-v3.lock"
BOOTSTRAP_PATH: Final = ANCHOR_ROOT / BOOTSTRAP_NAME

ANCHOR_SCHEMA_VERSION: Final = "riley.rc3-gate-e-execution-anchor.v1"
CONFIG_SCHEMA_VERSION: Final = "riley.rc3-gate-e-sealed-no-action-core-config.v1"
PROTOCOL_SCHEMA_VERSION: Final = "riley.rc3-gate-e-sealed-no-action-protocol.v1"
CORE_SCOPE: Final = "sealed-no-action-core"
CORE_AUTHORITY: Final = "sealed-no-action-protocol-only"

# This literal binds the future bootstrap to the reviewed v3 private-core
# source in addition to the root-owned manifest.  Any core update needs a new
# bootstrap review and a new literal rather than manifest-only substitution.
COMPILED_CORE_SHA256: Final = (
    "953de3d1cffa78d38317505b85334337293a54e30f946bf8e690913e6e75815c"
)
COMPILED_CORE_BYTE_LENGTH: Final = 25094

PARENT_LOCK_FD: Final = 7
CORE_FD: Final = 8
CONFIG_FD: Final = 9
CONTROL_FD: Final = 10
CHILD_SOURCE_MIN_FD: Final = 20

MAX_MANIFEST_BYTES: Final = 64 * 1024
MAX_CODE_BYTES: Final = 2 * 1024 * 1024
MAX_CONFIG_BYTES: Final = 8 * 1024
MAX_CONTROL_BYTES: Final = 4 * 1024
MAX_ENVIRONMENT_BYTES: Final = 64 * 1024
MAX_MOUNTINFO_BYTES: Final = 1024 * 1024
CONTROL_TIMEOUT_SECONDS: Final = 10.0
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_ANCHOR_FILESYSTEMS: Final = frozenset(("ext4", "xfs", "btrfs"))
REQUIRED_SEALS: Final = (
    fcntl.F_SEAL_SEAL
    | fcntl.F_SEAL_SHRINK
    | fcntl.F_SEAL_GROW
    | fcntl.F_SEAL_WRITE
)
PR_SET_PDEATHSIG: Final = 1
TERMINATING_SIGNALS: Final = frozenset(
    (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
)
INITIAL_IDENTITY_MAP_FIELDS: Final = (b"0", b"0", b"4294967295")


class BootstrapError(ValueError):
    """The root-installed no-action bootstrap contract is not safe."""


class ParentSignal(BootstrapError):
    """The bootstrap received a terminating signal while its child was alive."""


def _fail(message: str) -> NoReturn:
    raise BootstrapError(message)


def _die(message: str) -> NoReturn:
    print(f"RC3 Gate E v3 bootstrap: {message}", file=sys.stderr)
    raise SystemExit(2)


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
        _fail(f"cannot encode canonical JSON: {error}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("JSON object contains a duplicate key")
        result[key] = value
    return result


def _parse_canonical_object(
    raw: bytes,
    label: str,
    *,
    newline_terminated: bool,
) -> dict[str, object]:
    if newline_terminated:
        if not raw.endswith(b"\n"):
            _fail(f"{label} must end in exactly one newline")
        encoded = raw[:-1]
    else:
        encoded = raw
    if encoded.endswith(b"\n"):
        _fail(f"{label} has an unexpected extra newline")
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"cannot parse {label}: {error}")
    if type(value) is not dict or _canonical_json_bytes(value) != encoded:
        _fail(f"{label} is not a canonical JSON object")
    return value


def _sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb", buffering=0) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        _fail(f"cannot hash reviewed Python: {error}")
    return digest.hexdigest()


def _require_public_interpreter_and_invocation(arguments: Sequence[str]) -> None:
    if arguments != ["--bootstrap-core-smoke-test"]:
        _fail("only --bootstrap-core-smoke-test is accepted")
    if sys.argv[0] != os.fspath(BOOTSTRAP_PATH) or __file__ != os.fspath(BOOTSTRAP_PATH):
        _fail("must be invoked only from the fixed root-installed bootstrap path")
    if not (
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.ignore_environment
        and sys.dont_write_bytecode
        and sys.flags.optimize == 0
    ):
        _fail("must run under /usr/bin/python3.10 -I -S -E -B")
    expected_argv = [
        PYTHON_PATH,
        "-I",
        "-S",
        "-E",
        "-B",
        os.fspath(BOOTSTRAP_PATH),
        "--bootstrap-core-smoke-test",
    ]
    if getattr(sys, "orig_argv", None) != expected_argv:
        _fail("reviewed Python invocation arguments differ from the fixed contract")
    try:
        same_executable = os.path.samefile("/proc/self/exe", PYTHON_PATH)
        metadata = os.stat(PYTHON_PATH, follow_symlinks=False)
    except OSError as error:
        _fail(f"cannot authenticate reviewed Python: {error}")
    if (
        not same_executable
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        or _sha256_path(PYTHON_PATH) != PYTHON_SHA256
    ):
        _fail("reviewed Python executable differs from the pinned contract")


def _require_initial_identity_map(raw: bytes, label: str) -> None:
    lines = raw.splitlines()
    if len(lines) != 1 or lines[0].split() != list(INITIAL_IDENTITY_MAP_FIELDS):
        _fail(
            f"{label} must be the single full initial identity mapping "
            "'0 0 4294967295'"
        )


def _require_public_root_context() -> None:
    if os.getuid() != 0 or os.geteuid() != 0:
        _fail("public bootstrap requires a root-owned service or launcher; no privilege elevation exists here")
    try:
        inherited_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        inherited_sigchld = signal.getsignal(signal.SIGCHLD)
    except (AttributeError, OSError, ValueError) as error:
        _fail(f"cannot authenticate inherited signal state: {error}")
    if inherited_mask & TERMINATING_SIGNALS:
        _fail("public bootstrap requires HUP, INT, and TERM to start unblocked")
    if inherited_sigchld != signal.SIG_DFL:
        _fail("public bootstrap requires the default SIGCHLD disposition")
    for namespace in ("mnt", "user"):
        try:
            same_namespace = os.path.samefile(
                f"/proc/self/ns/{namespace}",
                f"/proc/1/ns/{namespace}",
            )
        except OSError as error:
            _fail(f"cannot authenticate host {namespace} namespace: {error}")
        if not same_namespace:
            _fail(f"bootstrap is outside PID 1's {namespace} namespace")
    try:
        same_root = os.path.samefile("/proc/self/root", "/proc/1/root")
        self_uid_map = Path("/proc/self/uid_map").read_bytes()
        init_uid_map = Path("/proc/1/uid_map").read_bytes()
        self_gid_map = Path("/proc/self/gid_map").read_bytes()
        init_gid_map = Path("/proc/1/gid_map").read_bytes()
    except OSError as error:
        _fail(f"cannot authenticate host root or namespace maps: {error}")
    if not same_root or self_uid_map != init_uid_map or self_gid_map != init_gid_map:
        _fail("bootstrap is outside PID 1's root or identity-map context")
    _require_initial_identity_map(self_uid_map, "self uid_map")
    _require_initial_identity_map(self_gid_map, "self gid_map")
    _require_initial_identity_map(init_uid_map, "PID 1 uid_map")
    _require_initial_identity_map(init_gid_map, "PID 1 gid_map")
    try:
        with open("/proc/self/environ", "rb", buffering=0) as handle:
            environment = handle.read(MAX_ENVIRONMENT_BYTES + 1)
    except OSError as error:
        _fail(f"cannot inspect bootstrap environment: {error}")
    if len(environment) > MAX_ENVIRONMENT_BYTES or environment:
        _fail("public bootstrap requires an empty execve environment")
    # CPython may synthesize LC_CTYPE in os.environ even when execve() received
    # an empty environment.  The raw procfs record above is the authority.
    descriptors = _live_descriptors()
    if descriptors != {0, 1, 2}:
        _fail(
            "public bootstrap must start with only stdio descriptors, got "
            f"{sorted(descriptors)!r}"
        )


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


def _safe_open_flags() -> tuple[int, int, int, int]:
    values: list[int] = []
    for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC", "O_NONBLOCK"):
        value = getattr(os, name, None)
        if type(value) is not int or value <= 0:
            _fail(f"platform lacks required os.{name}")
        values.append(value)
    return values[0], values[1], values[2], values[3]


def _require_no_posix_acl(descriptor: int, label: str) -> None:
    try:
        names = os.listxattr(descriptor)
    except (AttributeError, OSError) as error:
        _fail(f"cannot establish ACL policy for {label}: {error}")
    for name in names:
        if type(name) is not str:
            _fail(f"ACL name for {label} is not text")
        if name.startswith("system.posix_acl_"):
            _fail(f"{label} carries forbidden POSIX ACL {name!r}")


def _require_approved_local_filesystem(descriptor: int, label: str) -> None:
    """Reject network, overlay, and unknown ACL models for public anchors.

    Mode/UID/GID and POSIX xattr checks do not fully model NFSv4/CIFS/rich ACL
    policy.  The installed public path therefore accepts only a reviewed local
    filesystem type, determined from the held directory descriptor's device.
    """

    try:
        metadata = os.fstat(descriptor)
        device = f"{os.major(metadata.st_dev)}:{os.minor(metadata.st_dev)}"
        with open("/proc/self/mountinfo", "rb", buffering=0) as handle:
            raw = handle.read(MAX_MOUNTINFO_BYTES + 1)
    except OSError as error:
        _fail(f"cannot establish local filesystem policy for {label}: {error}")
    if not raw or len(raw) > MAX_MOUNTINFO_BYTES:
        _fail(f"cannot establish bounded mount policy for {label}")
    filesystem_types: set[str] = set()
    for line in raw.splitlines():
        fields = line.split()
        try:
            separator = fields.index(b"-")
        except ValueError:
            continue
        if len(fields) < 7 or separator + 1 >= len(fields) or fields[2] != device.encode("ascii"):
            continue
        try:
            filesystem = fields[separator + 1].decode("ascii")
        except UnicodeDecodeError:
            _fail(f"mount policy contains a non-ASCII filesystem name for {label}")
        if not filesystem or any(character.isspace() for character in filesystem):
            _fail(f"mount policy contains an unsafe filesystem name for {label}")
        filesystem_types.add(filesystem)
    if not filesystem_types:
        _fail(f"cannot find held filesystem device {device!r} for {label}")
    if not filesystem_types <= ALLOWED_ANCHOR_FILESYSTEMS:
        _fail(
            f"{label} is on an unapproved filesystem "
            f"{sorted(filesystem_types)!r}; expected only local "
            f"{sorted(ALLOWED_ANCHOR_FILESYSTEMS)!r}"
        )


def _require_directory(
    metadata: os.stat_result,
    descriptor: int,
    label: str,
    *,
    owner_uid: int,
    owner_gid: int,
    exact_mode: int | None = None,
) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        _fail(f"{label} must be a directory")
    if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
        _fail(f"{label} must be owned by UID {owner_uid} and GID {owner_gid}")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        _fail(f"{label} must not be group/world writable")
    if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
        _fail(f"{label} must not carry special mode bits")
    if exact_mode is not None and stat.S_IMODE(metadata.st_mode) != exact_mode:
        _fail(f"{label} must have mode {exact_mode:04o}")
    _require_no_posix_acl(descriptor, label)


def _require_regular(
    metadata: os.stat_result,
    descriptor: int,
    label: str,
    *,
    owner_uid: int,
    owner_gid: int,
    exact_mode: int,
    maximum_bytes: int,
    minimum_bytes: int,
) -> None:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        _fail(f"{label} must be a single-link regular file")
    if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
        _fail(f"{label} must be owned by UID {owner_uid} and GID {owner_gid}")
    if stat.S_IMODE(metadata.st_mode) != exact_mode:
        _fail(f"{label} must have mode {exact_mode:04o}")
    if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
        _fail(f"{label} must not carry special mode bits")
    if metadata.st_size < minimum_bytes or metadata.st_size > maximum_bytes:
        _fail(f"{label} has an unsafe byte length")
    _require_no_posix_acl(descriptor, label)


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
        _fail(f"{label} must be a normalized non-root absolute path")
    components = tuple(part for part in raw.split(os.path.sep) if part)
    if not components or any(part in {".", ".."} for part in components):
        _fail(f"{label} contains unsafe path components")
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
        _fail(f"{label} must be a normalized absolute path")
    return tuple(part for part in raw.split(os.path.sep) if part)


def _open_absolute_directory(
    path: Path,
    label: str,
    *,
    owner_uid: int,
    owner_gid: int,
    final_mode: int,
    trusted_prefix: Path = Path("/"),
    require_local_filesystem: bool = False,
) -> int:
    nofollow, directory, cloexec, _nonblock = _safe_open_flags()
    flags = os.O_RDONLY | nofollow | directory | cloexec
    components = _absolute_components(path, label)
    prefix = _prefix_components(trusted_prefix, f"{label} trusted prefix")
    if components[: len(prefix)] != prefix:
        _fail(f"{label} is outside its trusted prefix")
    try:
        current_fd = os.open(os.path.sep, flags)
    except OSError as error:
        _fail(f"cannot open filesystem root for {label}: {error}")
    try:
        root_metadata = os.fstat(current_fd)
        if require_local_filesystem:
            _require_approved_local_filesystem(current_fd, "filesystem root")
        if not prefix:
            _require_directory(
                root_metadata,
                current_fd,
                "filesystem root",
                owner_uid=owner_uid,
                owner_gid=owner_gid,
            )
        for index, component in enumerate(components):
            current_label = f"{label} component {component!r}"
            try:
                before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            except OSError as error:
                _fail(f"cannot inspect {current_label}: {error}")
            if index >= len(prefix):
                if not stat.S_ISDIR(before.st_mode):
                    _fail(f"{current_label} must be a directory")
            elif not stat.S_ISDIR(before.st_mode):
                _fail(f"{current_label} must be a directory")
            try:
                child_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as error:
                _fail(f"cannot open {current_label}: {error}")
            try:
                opened = os.fstat(child_fd)
                if require_local_filesystem:
                    _require_approved_local_filesystem(child_fd, current_label)
                if index >= len(prefix):
                    _require_directory(
                        opened,
                        child_fd,
                        current_label,
                        owner_uid=owner_uid,
                        owner_gid=owner_gid,
                        exact_mode=final_mode if index == len(components) - 1 else None,
                    )
                elif not stat.S_ISDIR(opened.st_mode):
                    _fail(f"{current_label} must be a directory")
                if _stable_identity(opened) != _stable_identity(before):
                    _fail(f"{current_label} changed while opening")
            except BaseException:
                os.close(child_fd)
                raise
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _relocate_fd(descriptor: int, minimum: int = CHILD_SOURCE_MIN_FD) -> int:
    if descriptor >= minimum:
        return descriptor
    try:
        relocated = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, minimum)
    except OSError as error:
        _fail(f"cannot relocate held descriptor: {error}")
    try:
        os.close(descriptor)
    except OSError as error:
        os.close(relocated)
        _fail(f"cannot close pre-relocation descriptor: {error}")
    return relocated


@dataclass(frozen=True)
class BoundFile:
    filename: str
    descriptor: int
    sha256: str
    byte_length: int
    device: int
    inode: int
    payload: bytes


def _read_hashed_regular(
    directory_fd: int,
    name: str,
    label: str,
    *,
    owner_uid: int,
    owner_gid: int,
    exact_mode: int,
    maximum_bytes: int,
    minimum_bytes: int,
    require_local_filesystem: bool = False,
) -> BoundFile:
    if type(name) is not str or not name or os.path.basename(name) != name:
        _fail(f"{label} must have a direct filename")
    nofollow, _directory, cloexec, nonblock = _safe_open_flags()
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        _fail(f"cannot inspect {label}: {error}")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | nofollow | nonblock | cloexec,
            dir_fd=directory_fd,
        )
    except OSError as error:
        _fail(f"cannot open {label} without following links: {error}")
    try:
        opened = os.fstat(descriptor)
        _require_regular(
            opened,
            descriptor,
            label,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            exact_mode=exact_mode,
            maximum_bytes=maximum_bytes,
            minimum_bytes=minimum_bytes,
        )
        if require_local_filesystem:
            _require_approved_local_filesystem(descriptor, label)
        if _stable_identity(opened) != _stable_identity(before):
            _fail(f"{label} changed while opening")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        seen = 0
        while True:
            try:
                block = os.read(descriptor, 64 * 1024)
            except OSError as error:
                _fail(f"cannot read {label}: {error}")
            if not block:
                break
            seen += len(block)
            if seen > maximum_bytes:
                _fail(f"{label} exceeded its byte limit")
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        if _stable_identity(after) != _stable_identity(opened) or seen != opened.st_size:
            _fail(f"{label} changed while hashing")
        try:
            named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            _fail(f"cannot re-inspect {label}: {error}")
        if _stable_identity(named_after) != _stable_identity(before):
            _fail(f"{label} changed while hashing")
        descriptor = _relocate_fd(descriptor)
        return BoundFile(
            filename=name,
            descriptor=descriptor,
            sha256=digest.hexdigest(),
            byte_length=seen,
            device=opened.st_dev,
            inode=opened.st_ino,
            payload=b"".join(chunks),
        )
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _manifest_file_spec(
    manifest: dict[str, object],
    field: str,
    expected_name: str,
) -> tuple[str, int]:
    value = manifest.get(field)
    if type(value) is not dict or set(value) != {"filename", "sha256", "byte_length"}:
        _fail(f"manifest {field} has an invalid shape")
    filename = value["filename"]
    digest = value["sha256"]
    byte_length = value["byte_length"]
    if filename != expected_name:
        _fail(f"manifest {field}.filename differs")
    if type(digest) is not str or SHA256_RE.fullmatch(digest) is None or digest == "0" * 64:
        _fail(f"manifest {field}.sha256 is unsafe")
    if type(byte_length) is not int or byte_length <= 0 or byte_length > MAX_CODE_BYTES:
        _fail(f"manifest {field}.byte_length is unsafe")
    return digest, byte_length


def _open_lock_file(
    directory_fd: int,
    *,
    owner_uid: int,
    owner_gid: int,
    require_local_filesystem: bool = False,
) -> int:
    nofollow, _directory, cloexec, nonblock = _safe_open_flags()
    try:
        before = os.stat(LOCK_NAME, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        _fail(f"cannot inspect preprovisioned lock: {error}")
    try:
        descriptor = os.open(
            LOCK_NAME,
            os.O_RDONLY | nofollow | nonblock | cloexec,
            dir_fd=directory_fd,
        )
    except OSError as error:
        _fail(f"cannot open preprovisioned lock without following links: {error}")
    try:
        opened = os.fstat(descriptor)
        _require_regular(
            opened,
            descriptor,
            "preprovisioned Gate E v3 lock",
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            exact_mode=0o600,
            maximum_bytes=0,
            minimum_bytes=0,
        )
        if require_local_filesystem:
            _require_approved_local_filesystem(
                descriptor,
                "preprovisioned Gate E v3 lock",
            )
        if _stable_identity(opened) != _stable_identity(before):
            _fail("preprovisioned lock changed while opening")
        try:
            named_after = os.stat(LOCK_NAME, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            _fail(f"cannot re-inspect preprovisioned lock: {error}")
        if _stable_identity(named_after) != _stable_identity(before):
            _fail("preprovisioned lock changed while opening")
        return _relocate_fd(descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


@dataclass
class AnchorBundle:
    anchor_root_fd: int
    bootstrap: BoundFile
    core: BoundFile
    lock_fd: int
    lock_path: Path

    def close_non_lock(self, *, require_success: bool = False) -> None:
        failures: list[str] = []
        for label, descriptor in (
            ("anchor root", self.anchor_root_fd),
            ("bootstrap source", self.bootstrap.descriptor),
            ("private-core source", self.core.descriptor),
        ):
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except OSError as error:
                failures.append(f"{label} FD {descriptor}: {error}")
        if failures and require_success:
            _fail("cannot close parent source descriptors before handshake: " + "; ".join(failures))
        self.anchor_root_fd = -1
        self.bootstrap = BoundFile("", -1, "", 0, 0, 0, b"")
        self.core = BoundFile("", -1, "", 0, 0, 0, b"")

    def close(self) -> None:
        self.close_non_lock()
        try:
            os.close(self.lock_fd)
        except OSError:
            pass
        self.lock_fd = -1


def _verify_anchor(
    anchor_root: Path,
    lock_directory: Path,
    *,
    owner_uid: int,
    owner_gid: int,
    trusted_prefix: Path = Path("/"),
    require_local_filesystem: bool = False,
) -> AnchorBundle:
    root_fd = _open_absolute_directory(
        anchor_root,
        "execution anchor root",
        owner_uid=owner_uid,
        owner_gid=owner_gid,
        final_mode=0o755,
        trusted_prefix=trusted_prefix,
        require_local_filesystem=require_local_filesystem,
    )
    root_fd = _relocate_fd(root_fd)
    bootstrap: BoundFile | None = None
    core: BoundFile | None = None
    lock_fd: int | None = None
    try:
        manifest = _read_hashed_regular(
            root_fd,
            MANIFEST_NAME,
            "execution anchor manifest",
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            exact_mode=0o644,
            maximum_bytes=MAX_MANIFEST_BYTES,
            minimum_bytes=1,
            require_local_filesystem=require_local_filesystem,
        )
        try:
            manifest_value = _parse_canonical_object(
                manifest.payload,
                "execution anchor manifest",
                newline_terminated=True,
            )
        finally:
            os.close(manifest.descriptor)
        if set(manifest_value) != {"schema_version", "bootstrap", "core", "lock_directory"}:
            _fail("execution anchor manifest has an unexpected field set")
        if manifest_value["schema_version"] != ANCHOR_SCHEMA_VERSION:
            _fail("execution anchor manifest schema_version differs")
        if manifest_value["lock_directory"] != os.fspath(lock_directory):
            _fail("execution anchor manifest lock_directory differs")
        expected_bootstrap_sha, expected_bootstrap_length = _manifest_file_spec(
            manifest_value,
            "bootstrap",
            BOOTSTRAP_NAME,
        )
        expected_core_sha, expected_core_length = _manifest_file_spec(
            manifest_value,
            "core",
            CORE_NAME,
        )
        bootstrap = _read_hashed_regular(
            root_fd,
            BOOTSTRAP_NAME,
            "execution anchor bootstrap",
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            exact_mode=0o755,
            maximum_bytes=MAX_CODE_BYTES,
            minimum_bytes=1,
            require_local_filesystem=require_local_filesystem,
        )
        core = _read_hashed_regular(
            root_fd,
            CORE_NAME,
            "execution anchor private core",
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            exact_mode=0o644,
            maximum_bytes=MAX_CODE_BYTES,
            minimum_bytes=1,
            require_local_filesystem=require_local_filesystem,
        )
        if (
            bootstrap.sha256 != expected_bootstrap_sha
            or bootstrap.byte_length != expected_bootstrap_length
        ):
            _fail("root-owned manifest bootstrap digest differs")
        if core.sha256 != expected_core_sha or core.byte_length != expected_core_length:
            _fail("root-owned manifest core digest differs")
        if (
            core.sha256 != COMPILED_CORE_SHA256
            or core.byte_length != COMPILED_CORE_BYTE_LENGTH
        ):
            _fail("private core differs from the bootstrap's compiled review pin")
        lock_directory_fd = _open_absolute_directory(
            lock_directory,
            "execution anchor lock directory",
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            final_mode=0o700,
            trusted_prefix=trusted_prefix,
            require_local_filesystem=require_local_filesystem,
        )
        try:
            lock_fd = _open_lock_file(
                lock_directory_fd,
                owner_uid=owner_uid,
                owner_gid=owner_gid,
                require_local_filesystem=require_local_filesystem,
            )
        finally:
            os.close(lock_directory_fd)
        return AnchorBundle(
            anchor_root_fd=root_fd,
            bootstrap=bootstrap,
            core=core,
            lock_fd=lock_fd,
            lock_path=lock_directory / LOCK_NAME,
        )
    except BaseException:
        for descriptor in (
            root_fd,
            bootstrap.descriptor if bootstrap is not None else -1,
            core.descriptor if core is not None else -1,
            lock_fd if lock_fd is not None else -1,
        ):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        raise


def _require_running_bootstrap_matches(bundle: AnchorBundle, runtime_path: Path) -> None:
    expected = os.fspath(runtime_path)
    if __file__ != expected:
        _fail("loaded bootstrap source path differs from the authenticated anchor leaf")
    try:
        named = os.stat(expected, follow_symlinks=False)
        opened = os.fstat(bundle.bootstrap.descriptor)
    except OSError as error:
        _fail(f"cannot compare running bootstrap to held anchor leaf: {error}")
    if _stable_identity(named) != _stable_identity(opened):
        _fail("running bootstrap source differs from the held root-owned anchor leaf")


def _proc_starttime_ticks(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="ascii") as handle:
            raw = handle.read()
    except OSError as error:
        _fail(f"cannot read process start time: {error}")
    delimiter = raw.rfind(") ")
    if delimiter < 0:
        _fail("process stat has an unexpected shape")
    fields = raw[delimiter + 2 :].split()
    if len(fields) <= 19 or not fields[19].isdigit():
        _fail("process start time is invalid")
    value = int(fields[19])
    if value <= 0:
        _fail("process start time is invalid")
    return value


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(descriptor, payload[offset:])
        except OSError as error:
            _fail(f"cannot write sealed anonymous bytes: {error}")
        if written <= 0:
            _fail("short write while sealing anonymous bytes")
        offset += written


def _sealed_memfd(label: str, payload: bytes, maximum_bytes: int) -> int:
    if not payload or len(payload) > maximum_bytes:
        _fail(f"{label} has an unsafe byte length")
    try:
        descriptor = os.memfd_create(
            label,
            os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
        )
    except (AttributeError, OSError) as error:
        _fail(f"cannot create sealed {label}: {error}")
    try:
        _write_all(descriptor, payload)
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, REQUIRED_SEALS)
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != REQUIRED_SEALS:
            _fail(f"sealed {label} does not retain every required seal")
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 0
            or metadata.st_size != len(payload)
        ):
            _fail(f"sealed {label} has an unexpected descriptor shape")
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _is_socketpair_name(value: object) -> bool:
    return (
        value in {"", b"", None}
        or (
            type(value) is bytes
            and re.fullmatch(br"\x00[0-9a-f]{5}", value) is not None
        )
    )


def _close_received_rights(ancillary: list[tuple[int, int, bytes]]) -> None:
    descriptor_size = struct.calcsize("i")
    for level, kind, payload in ancillary:
        if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
            continue
        if len(payload) % descriptor_size:
            continue
        for descriptor in struct.unpack(f"{len(payload) // descriptor_size}i", payload):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _receive_child_packet(
    control: socket.socket,
    *,
    child_pid: int,
    child_uid: int,
    child_gid: int,
) -> dict[str, object]:
    credential_size = struct.calcsize("3i")
    try:
        payload, ancillary, flags, address = control.recvmsg(
            MAX_CONTROL_BYTES,
            socket.CMSG_SPACE(credential_size),
        )
    except (OSError, socket.timeout) as error:
        _fail(f"cannot receive private-core control packet: {error}")
    _close_received_rights(ancillary)
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
        _fail("private-core control packet was truncated")
    if not _is_socketpair_name(address):
        _fail("private-core control packet has a non-socketpair sender name")
    if not payload:
        _fail("private-core control packet is empty")
    if len(ancillary) != 1:
        _fail("private-core control packet must carry exactly one credential record")
    level, kind, credential_raw = ancillary[0]
    if (
        level != socket.SOL_SOCKET
        or kind != socket.SCM_CREDENTIALS
        or len(credential_raw) != credential_size
    ):
        _fail("private-core control packet carries an unexpected ancillary record")
    if struct.unpack("3i", credential_raw) != (child_pid, child_uid, child_gid):
        _fail("private-core control packet credentials differ from the forked child")
    return _parse_canonical_object(
        payload,
        "private-core control packet",
        newline_terminated=False,
    )


def _send_parent_packet(control: socket.socket, packet: dict[str, object]) -> None:
    payload = _canonical_json_bytes(packet)
    if not payload or len(payload) > MAX_CONTROL_BYTES:
        _fail("parent control packet exceeds the fixed byte bound")
    try:
        sent = control.send(payload)
    except OSError as error:
        _fail(f"cannot send private-core control packet: {error}")
    if sent != len(payload):
        _fail("parent control packet was not sent as one complete record")


def _require_child_packet(
    packet: dict[str, object],
    expected_kind: str,
    nonce: str,
    core_sha256: str,
    config_sha256: str,
) -> None:
    expected_fields = {
        "schema_version",
        "kind",
        "nonce",
        "core_sha256",
        "config_sha256",
    }
    if expected_kind == "complete":
        expected_fields.add("guarantees")
    if set(packet) != expected_fields:
        _fail("private-core control packet has an unexpected field set")
    if (
        packet["schema_version"] != PROTOCOL_SCHEMA_VERSION
        or packet["kind"] != expected_kind
        or packet["nonce"] != nonce
        or packet["core_sha256"] != core_sha256
        or packet["config_sha256"] != config_sha256
    ):
        _fail("private-core control packet does not match the sealed session")
    if expected_kind == "complete" and packet["guarantees"] != _no_action_guarantees():
        _fail("private-core completion claims an unsafe capability")


def _no_action_guarantees() -> dict[str, bool]:
    return {
        "gpu_lock_acquired": False,
        "gpu_queried": False,
        "docker_invoked": False,
        "evidence_created": False,
        "semantic_replay_run": False,
        "receipt_published": False,
        "qualification_decided": False,
    }


def _live_descriptors() -> set[int]:
    try:
        scanner_fd = os.open(
            "/proc/self/fd",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
    except OSError as error:
        _fail(f"cannot open inherited-descriptor view: {error}")
    try:
        names = os.listdir(scanner_fd)
    except OSError as error:
        os.close(scanner_fd)
        _fail(f"cannot enumerate inherited descriptors: {error}")
    finally:
        try:
            os.close(scanner_fd)
        except OSError:
            pass
    descriptors: set[int] = set()
    for name in names:
        try:
            descriptor = int(name)
        except ValueError:
            _fail("inherited-descriptor view contains a non-numeric name")
        if descriptor == scanner_fd:
            continue
        try:
            os.fstat(descriptor)
        except OSError as error:
            if error.errno == errno.EBADF:
                continue
            _fail(f"cannot inspect inherited descriptor {descriptor}: {error}")
        descriptors.add(descriptor)
    return descriptors


def _close_all_except(allowed: set[int]) -> None:
    for descriptor in _live_descriptors():
        if descriptor in allowed:
            continue
        try:
            os.close(descriptor)
        except OSError as error:
            _fail(f"cannot close child-inherited descriptor {descriptor}: {error}")
    remaining = _live_descriptors()
    if remaining != allowed:
        _fail(
            "child descriptor cleanup did not retain exactly "
            f"{sorted(allowed)!r}, got {sorted(remaining)!r}"
        )


def _set_child_parent_death_signal(parent_pid: int) -> None:
    try:
        # The parent blocks these before fork to eliminate the handler-install
        # gap.  The private core must not inherit that policy: HUP and INT are
        # reset here, while the sealed core resets TERM again after exec.
        for signum in TERMINATING_SIGNALS:
            signal.signal(signum, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, TERMINATING_SIGNALS)
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except (AttributeError, OSError, ValueError) as error:
        _fail(f"cannot configure child parent-death signal: {error}")
    if result != 0:
        _fail(f"cannot configure child parent-death signal: errno={ctypes.get_errno()}")
    if os.getppid() != parent_pid:
        _fail("bootstrap parent exited before the child exec handoff")


def _move_parent_lock_to_reserved_fd(bundle: AnchorBundle) -> None:
    if bundle.lock_fd == PARENT_LOCK_FD:
        try:
            fcntl.fcntl(bundle.lock_fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
        except OSError as error:
            _fail(f"cannot mark reserved parent lock close-on-exec: {error}")
        return
    try:
        os.fstat(PARENT_LOCK_FD)
    except OSError as error:
        if error.errno != errno.EBADF:
            _fail(f"cannot inspect reserved parent lock descriptor: {error}")
    else:
        _fail("reserved parent lock descriptor is already occupied")
    try:
        os.dup2(bundle.lock_fd, PARENT_LOCK_FD, inheritable=False)
        os.close(bundle.lock_fd)
    except OSError as error:
        _fail(f"cannot move parent lock to reserved descriptor: {error}")
    bundle.lock_fd = PARENT_LOCK_FD


def _acquire_parent_lock(bundle: AnchorBundle) -> None:
    _move_parent_lock_to_reserved_fd(bundle)
    try:
        fcntl.flock(bundle.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        _fail(f"preprovisioned parent-only lock is unavailable: {error}")
    try:
        metadata = os.fstat(bundle.lock_fd)
    except OSError as error:
        _fail(f"cannot re-inspect acquired parent lock: {error}")
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_size != 0
    ):
        _fail("acquired parent lock changed its fixed contract")


def _wait_child(
    child_pid: int,
    *,
    timeout_seconds: float | None = None,
) -> int:
    deadline = (
        None
        if timeout_seconds is None
        else time.monotonic() + timeout_seconds
    )
    while True:
        try:
            observed_pid, status = os.waitpid(
                child_pid,
                0 if deadline is None else os.WNOHANG,
            )
            if observed_pid == child_pid:
                return status
            if observed_pid != 0:
                _fail("unexpected child identifier while reaping private core")
        except InterruptedError:
            continue
        except OSError as error:
            _fail(f"cannot reap private-core child: {error}")
        if deadline is None:
            _fail("blocking private-core wait returned without a child status")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _fail("private-core child did not exit before the fixed timeout")
        time.sleep(min(0.02, remaining))


def _terminate_and_reap(child_pid: int) -> None:
    try:
        os.kill(child_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as error:
        _fail(f"cannot terminate private-core child: {error}")
    try:
        _wait_child(child_pid, timeout_seconds=CONTROL_TIMEOUT_SECONDS)
    except BootstrapError:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            _fail(f"cannot force-stop private-core child: {error}")
        _wait_child(child_pid, timeout_seconds=CONTROL_TIMEOUT_SECONDS)


class _ParentSignalForwarder:
    def __init__(self, child_pid: int, restore_mask: set[signal.Signals]) -> None:
        self.child_pid = child_pid
        self.restore_mask = restore_mask
        self.saved: dict[int, object] = {}

    def __enter__(self) -> "_ParentSignalForwarder":
        try:
            for signum in TERMINATING_SIGNALS:
                self.saved[signum] = signal.getsignal(signum)
                signal.signal(signum, self._forward_as_term)
            # A terminating signal that arrived between pthread_sigmask() and
            # fork is now delivered only after its forwarding handler exists.
            signal.pthread_sigmask(signal.SIG_SETMASK, self.restore_mask)
        except BaseException:
            try:
                signal.pthread_sigmask(signal.SIG_BLOCK, TERMINATING_SIGNALS)
            except (AttributeError, OSError, ValueError):
                pass
            self._restore_handlers()
            raise
        return self

    def _forward_as_term(self, signum: int, _frame: object) -> NoReturn:
        try:
            os.kill(self.child_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            pass
        raise ParentSignal(f"bootstrap received terminating signal {signum}")

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        # Do not reopen the old-handler gap between the handshake and the
        # caller's final restoration of the pre-fork signal mask.
        try:
            signal.pthread_sigmask(signal.SIG_BLOCK, TERMINATING_SIGNALS)
        finally:
            self._restore_handlers()

    def _restore_handlers(self) -> None:
        for signum, handler in self.saved.items():
            signal.signal(signum, handler)
        self.saved.clear()


def _fork_exec_private_core(
    core_memfd: int,
    config_memfd: int,
    child_control: socket.socket,
    parent_control: socket.socket,
) -> int:
    try:
        core_source_fd = fcntl.fcntl(core_memfd, fcntl.F_DUPFD_CLOEXEC, CHILD_SOURCE_MIN_FD)
        config_source_fd = fcntl.fcntl(config_memfd, fcntl.F_DUPFD_CLOEXEC, CHILD_SOURCE_MIN_FD)
        control_source_fd = fcntl.fcntl(
            child_control.fileno(),
            fcntl.F_DUPFD_CLOEXEC,
            CHILD_SOURCE_MIN_FD,
        )
    except OSError as error:
        _fail(f"cannot duplicate sealed child handoff descriptors: {error}")
    try:
        child_pid = os.fork()
    except OSError as error:
        for descriptor in (core_source_fd, config_source_fd, control_source_fd):
            os.close(descriptor)
        _fail(f"cannot fork private no-action core: {error}")
    if child_pid != 0:
        for descriptor in (core_source_fd, config_source_fd, control_source_fd):
            os.close(descriptor)
        return child_pid
    try:
        parent_pid = os.getppid()
        parent_control.close()
        child_control.close()
        os.dup2(core_source_fd, CORE_FD, inheritable=True)
        os.dup2(config_source_fd, CONFIG_FD, inheritable=True)
        os.dup2(control_source_fd, CONTROL_FD, inheritable=True)
        _close_all_except({0, 1, 2, CORE_FD, CONFIG_FD, CONTROL_FD})
        # The core's exact descriptor allowlist includes stdio.  A service
        # launcher may mark its log streams CLOEXEC, so preserve all three
        # explicit stdio descriptors rather than failing later at exec.
        for descriptor in (0, 1, 2):
            os.set_inheritable(descriptor, True)
        _set_child_parent_death_signal(parent_pid)
        os.execve(
            PYTHON_PATH,
            [
                PYTHON_PATH,
                "-I",
                "-S",
                "-E",
                "-B",
                "/proc/self/fd/8",
                "--sealed-no-action-core",
            ],
            {},
        )
    except BaseException as error:
        try:
            os.write(2, f"RC3 Gate E v3 bootstrap child handoff failed: {error}\n".encode("utf-8"))
        except OSError:
            pass
    os._exit(127)


def _run_no_action_bundle(bundle: AnchorBundle) -> dict[str, object]:
    _acquire_parent_lock(bundle)
    core_sha256 = bundle.core.sha256
    core_byte_length = bundle.core.byte_length
    core_memfd = -1
    config_memfd = -1
    parent_control: socket.socket | None = None
    child_control: socket.socket | None = None
    try:
        core_memfd = _sealed_memfd(
            "riley-rc3-gate-e-v3-core",
            bundle.core.payload,
            MAX_CODE_BYTES,
        )
        nonce = os.urandom(32).hex()
        config = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "scope": CORE_SCOPE,
            "authority": CORE_AUTHORITY,
            "nonce": nonce,
            "parent_pid": os.getpid(),
            "parent_starttime_ticks": _proc_starttime_ticks(os.getpid()),
            "parent_uid": os.getuid(),
            "parent_gid": os.getgid(),
            "parent_executable": PYTHON_PATH,
            "core_fd": CORE_FD,
            "core_sha256": core_sha256,
            "core_byte_length": core_byte_length,
            "config_fd": CONFIG_FD,
            "control_fd": CONTROL_FD,
        }
        config_raw = _canonical_json_bytes(config) + b"\n"
        config_memfd = _sealed_memfd(
            "riley-rc3-gate-e-v3-config",
            config_raw,
            MAX_CONFIG_BYTES,
        )
        config_sha256 = hashlib.sha256(config_raw).hexdigest()
        parent_control, child_control = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC,
        )
        parent_control.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        child_control.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        parent_control.settimeout(CONTROL_TIMEOUT_SECONDS)
        try:
            original_signal_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                TERMINATING_SIGNALS,
            )
        except (AttributeError, OSError, ValueError) as error:
            _fail(f"cannot block terminating signals before fork: {error}")
        child_pid: int | None = None
        child_reaped = False
        child_pid = _fork_exec_private_core(
            core_memfd,
            config_memfd,
            child_control,
            parent_control,
        )
        child_control.close()
        child_control = None
        os.close(core_memfd)
        core_memfd = -1
        os.close(config_memfd)
        config_memfd = -1
        # The parent keeps only the lock and control endpoint after fork; all
        # anchor/source descriptors are explicitly closed before the handshake.
        bundle.close_non_lock(require_success=True)
        with _ParentSignalForwarder(child_pid, original_signal_mask):
            _send_parent_packet(
                parent_control,
                {
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                    "kind": "init",
                    "nonce": nonce,
                    "config_sha256": config_sha256,
                },
            )
            ready = _receive_child_packet(
                parent_control,
                child_pid=child_pid,
                child_uid=os.getuid(),
                child_gid=os.getgid(),
            )
            _require_child_packet(
                ready,
                "ready",
                nonce,
                core_sha256,
                config_sha256,
            )
            _send_parent_packet(
                parent_control,
                {
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                    "kind": "run_no_action",
                    "nonce": nonce,
                    "config_sha256": config_sha256,
                },
            )
            complete = _receive_child_packet(
                parent_control,
                child_pid=child_pid,
                child_uid=os.getuid(),
                child_gid=os.getgid(),
            )
            _require_child_packet(
                complete,
                "complete",
                nonce,
                core_sha256,
                config_sha256,
            )
            status = _wait_child(
                child_pid,
                timeout_seconds=CONTROL_TIMEOUT_SECONDS,
            )
            child_reaped = True
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            _fail("private core did not exit successfully after no-action completion")
        return {
            "scope": "bootstrap-core-no-action-smoke-test-only",
            "status": "no-action-complete",
            "core_sha256": core_sha256,
            "config_sha256": config_sha256,
            "parent_lock_fd": PARENT_LOCK_FD,
            "guarantees": _no_action_guarantees(),
        }
    except BaseException:
        if "child_pid" in locals() and child_pid is not None and not child_reaped:
            _terminate_and_reap(child_pid)
        raise
    finally:
        if "original_signal_mask" in locals():
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, original_signal_mask)
            except (AttributeError, OSError, ValueError):
                pass
        for control in (parent_control, child_control):
            if control is None:
                continue
            try:
                control.close()
            except OSError:
                pass
        for descriptor in (core_memfd, config_memfd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _run_no_action_for_test(
    anchor_root: Path,
    lock_directory: Path,
    *,
    owner_uid: int,
    owner_gid: int,
    trusted_prefix: Path,
) -> dict[str, object]:
    """Private CPU-only seam; public CLI has no anchor/path override."""

    bundle = _verify_anchor(
        anchor_root,
        lock_directory,
        owner_uid=owner_uid,
        owner_gid=owner_gid,
        trusted_prefix=trusted_prefix,
    )
    try:
        _require_running_bootstrap_matches(bundle, anchor_root / BOOTSTRAP_NAME)
        return _run_no_action_bundle(bundle)
    finally:
        bundle.close()


def _run_fixed_no_action_smoke_test() -> None:
    bundle = _verify_anchor(
        ANCHOR_ROOT,
        LOCK_DIRECTORY,
        owner_uid=0,
        owner_gid=0,
        require_local_filesystem=True,
    )
    try:
        _require_running_bootstrap_matches(bundle, BOOTSTRAP_PATH)
        _run_no_action_bundle(bundle)
    finally:
        bundle.close()


def main(arguments: Sequence[str]) -> int:
    try:
        # This remains ahead of every anchor/lock/socket/fork operation so a
        # mutable checkout never becomes a callable bootstrap.
        _require_public_interpreter_and_invocation(arguments)
        _require_public_root_context()
        _run_fixed_no_action_smoke_test()
    except BootstrapError as error:
        _die(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
