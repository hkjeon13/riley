#!/usr/bin/python3.10
"""No-action source-binding foundation for a future RC3 Gate E producer.

This entrypoint deliberately accepts only a CPU-only source-contract probe.
It does not open the GPU lock, invoke Bash, start Docker, create evidence, or
make a Gate E decision.  Its purpose is to establish the non-Bash trust
boundary needed before a future source-locked performance child can exist.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import stat
import sys


PYTHON_PATH = "/usr/bin/python3.10"
PYTHON_SHA256 = "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
REMOTE_REPOSITORY_ROOT = "/home/psyche/rustinfer-vllm-roadmap-serial"
LAUNCHER_RELATIVE_PATH = "ci/release/run_remote_rc3_gate_e_session_v2.py"
PERFORMANCE_BODY_RELATIVE_PATH = "ci/run_remote_release_performance.sh"
PERFORMANCE_BODY_NAME = "run_remote_release_performance.sh"
SOURCE_SNAPSHOT_FD = 8
MAX_SOURCE_BYTES = 2 * 1024 * 1024
REQUIRED_SEALS = (
    fcntl.F_SEAL_SEAL
    | fcntl.F_SEAL_SHRINK
    | fcntl.F_SEAL_GROW
    | fcntl.F_SEAL_WRITE
)


def _die(message: str) -> None:
    print(f"RC3 Gate E session v2: {message}", file=sys.stderr)
    raise SystemExit(2)


def _sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_isolated_reviewed_python() -> None:
    if not (
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.ignore_environment
    ):
        _die("must run under /usr/bin/python3.10 -I -S -E")
    try:
        same_executable = os.path.samefile("/proc/self/exe", PYTHON_PATH)
    except OSError as error:
        _die(f"cannot authenticate Python executable: {error}")
    if not same_executable:
        _die("Python executable path differs from the reviewed interpreter")
    try:
        metadata = os.stat(PYTHON_PATH, follow_symlinks=False)
    except OSError as error:
        _die(f"cannot inspect reviewed Python: {error}")
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & 0o022
        or _sha256_path(PYTHON_PATH) != PYTHON_SHA256
    ):
        _die("reviewed Python bytes are not safe")


def _open_directory_at(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        _die(f"cannot open trusted directory component {name!r}: {error}")
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        _die(f"cannot inspect trusted directory component {name!r}: {error}")
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        os.close(descriptor)
        _die(f"trusted directory component {name!r} is unsafe")
    return descriptor


def _open_trusted_repository_root() -> tuple[int, dict[str, int | str]]:
    expected_launcher = f"{REMOTE_REPOSITORY_ROOT}/{LAUNCHER_RELATIVE_PATH}"
    if sys.argv[0] != expected_launcher or __file__ != expected_launcher:
        _die("must be invoked by the fixed trusted launcher path")
    root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        root_fd = os.open(REMOTE_REPOSITORY_ROOT, root_flags)
    except OSError as error:
        _die(f"cannot open fixed trusted repository root: {error}")
    try:
        root = os.fstat(root_fd)
        named_root = os.stat(REMOTE_REPOSITORY_ROOT, follow_symlinks=False)
        if (
            not stat.S_ISDIR(root.st_mode)
            or root.st_uid != os.getuid()
            or (root.st_dev, root.st_ino) != (named_root.st_dev, named_root.st_ino)
        ):
            _die("fixed trusted repository root changed while opening")
        ci_fd = _open_directory_at(root_fd, "ci")
        try:
            release_fd = _open_directory_at(ci_fd, "release")
            try:
                launcher_flags = (
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
                )
                launcher_fd = os.open(
                    "run_remote_rc3_gate_e_session_v2.py",
                    launcher_flags,
                    dir_fd=release_fd,
                )
                try:
                    loaded = os.stat(__file__, follow_symlinks=False)
                    opened = os.fstat(launcher_fd)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or opened.st_uid != os.getuid()
                        or (opened.st_dev, opened.st_ino)
                        != (loaded.st_dev, loaded.st_ino)
                    ):
                        _die("running launcher does not match the fixed trusted leaf")
                finally:
                    os.close(launcher_fd)
            finally:
                os.close(release_fd)
        finally:
            os.close(ci_fd)
        return root_fd, {
            "repository_root": REMOTE_REPOSITORY_ROOT,
            "repository_root_dev": root.st_dev,
            "repository_root_ino": root.st_ino,
        }
    except BaseException:
        os.close(root_fd)
        raise


def _require_fd_available(descriptor: int) -> None:
    try:
        os.fstat(descriptor)
    except OSError as error:
        if error.errno == errno.EBADF:
            return
        _die(f"cannot inspect reserved source descriptor: {error}")
    _die("reserved source descriptor is already open")


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            _die("short write while sealing source bytes")
        offset += written


def _open_sealed_body_snapshot(
    repository_root_fd: int,
    root_identity: dict[str, int | str],
) -> tuple[int, dict[str, object]]:
    _require_fd_available(SOURCE_SNAPSHOT_FD)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        ci_fd = _open_directory_at(repository_root_fd, "ci")
    except OSError as error:
        _die(f"cannot open fixed performance-body directory: {error}")
    try:
        try:
            input_fd = os.open(PERFORMANCE_BODY_NAME, flags, dir_fd=ci_fd)
        except OSError as error:
            _die(f"cannot open performance body without following links: {error}")
        opened = os.fstat(input_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
        ):
            _die("performance body is not a stable regular file")
        if opened.st_size <= 0 or opened.st_size > MAX_SOURCE_BYTES:
            _die("performance body exceeds the bounded source contract")
        try:
            snapshot_fd = os.memfd_create(
                "riley-rc3-gate-e-performance-body",
                os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
            )
        except (AttributeError, OSError) as error:
            _die(f"cannot create sealed source snapshot: {error}")
        try:
            digest = hashlib.sha256()
            retained = 0
            while True:
                block = os.read(input_fd, 64 * 1024)
                if not block:
                    break
                retained += len(block)
                if retained > MAX_SOURCE_BYTES:
                    _die("performance body exceeds the bounded source contract")
                digest.update(block)
                _write_all(snapshot_fd, block)
            if retained != opened.st_size:
                _die("performance body changed while its source bytes were read")
            os.lseek(snapshot_fd, 0, os.SEEK_SET)
            fcntl.fcntl(snapshot_fd, fcntl.F_ADD_SEALS, REQUIRED_SEALS)
            if fcntl.fcntl(snapshot_fd, fcntl.F_GET_SEALS) != REQUIRED_SEALS:
                _die("sealed source snapshot does not retain every required seal")
            snapshot = os.fstat(snapshot_fd)
            if not stat.S_ISREG(snapshot.st_mode) or snapshot.st_size != retained:
                _die("sealed source snapshot has an unexpected type or size")
            os.dup2(snapshot_fd, SOURCE_SNAPSHOT_FD, inheritable=False)
            if snapshot_fd != SOURCE_SNAPSHOT_FD:
                os.close(snapshot_fd)
            return SOURCE_SNAPSHOT_FD, {
                **root_identity,
                "body_path": PERFORMANCE_BODY_RELATIVE_PATH,
                "body_source_dev": opened.st_dev,
                "body_source_ino": opened.st_ino,
                "sealed_snapshot_dev": snapshot.st_dev,
                "sealed_snapshot_ino": snapshot.st_ino,
                "sealed_snapshot_sha256": digest.hexdigest(),
                "sealed_snapshot_size": retained,
            }
        except BaseException:
            try:
                os.close(snapshot_fd)
            except OSError:
                pass
            raise
    finally:
        try:
            os.close(input_fd)
        except UnboundLocalError:
            pass
        os.close(ci_fd)


def _usage(stream: object) -> None:
    print(
        "usage: /usr/bin/python3.10 -I -S -E "
        "/home/psyche/rustinfer-vllm-roadmap-serial/ci/release/"
        "run_remote_rc3_gate_e_session_v2.py "
        "--performance-source-contract-probe",
        file=stream,
    )
    print(
        "This is a no-action source-binding probe for a future RC3 Gate E "
        "producer. It does not open a GPU lock, invoke Bash, start Docker, "
        "create evidence, or publish a receipt.",
        file=stream,
    )


def main(arguments: list[str]) -> int:
    if arguments == ["--help"] or arguments == ["-h"]:
        _usage(sys.stdout)
        return 0
    if arguments != ["--performance-source-contract-probe"]:
        _usage(sys.stderr)
        return 2
    _require_isolated_reviewed_python()
    repository_root_fd, root_identity = _open_trusted_repository_root()
    try:
        descriptor, source = _open_sealed_body_snapshot(
            repository_root_fd,
            root_identity,
        )
        try:
            if descriptor != SOURCE_SNAPSHOT_FD:
                _die("sealed source snapshot did not occupy its reserved descriptor")
            payload = {
                "schema_version": "riley.rc3-gate-e-session-source-contract.v2",
                "status": "source-bound-no-action",
                "source": source,
                "guarantees": {
                    "gpu_lock_opened": False,
                    "bash_invoked": False,
                    "docker_invoked": False,
                    "evidence_created": False,
                    "receipt_published": False,
                    "qualification_decided": False,
                },
            }
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
        finally:
            try:
                os.close(descriptor)
            except UnboundLocalError:
                pass
            except OSError:
                pass
    finally:
        os.close(repository_root_fd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
