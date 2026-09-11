#!/usr/bin/env python3
"""Check a supplied CPython before reconstructed-runtime materialization.

This is a narrow readiness diagnostic for the reviewed PR16 replay, whose
``tomllib`` dependency cannot run on the remote host's ambient Python 3.10.
The controller itself never downloads or installs Python/uv, writes an
evidence file, invokes a materializer, or intentionally calls Docker/GPU
APIs. The output is transient ``checked/not-run`` stdout only; it is not a
receipt or qualification input.

The caller supplies one external absolute interpreter path.  This checker
opens that leaf through no-follow directory descriptors, hashes the held
binary, then invokes its held-descriptor path via ``/proc/self/fd`` with a
fixed isolated standard-library probe. Holding a descriptor prevents a path
swap, not a same-inode content mutation after hashing. This is therefore not a
sandbox or a trusted-runtime attestation: the operator must independently
trust the full runtime tree before allowing this checker to execute it. It does
not claim a future materializer receives the same descriptor or an immutable
runtime tree; those remain separate prerequisites.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import selectors
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence


sys.dont_write_bytecode = True


PREREQUISITE_VERSION = "riley.reconstructed-runtime-python-prerequisite.v1"
SCOPE = "reconstructed-runtime-interpreter-readiness-only"
AUTHORITY = "interpreter-readiness-only"
STATUS = "checked"
QUALIFICATION_STATUS = "not-run"

EXPECTED_IMPLEMENTATION = "cpython"
EXPECTED_VERSION = "3.13.15"
EXPECTED_PLATFORM = "linux"
EXPECTED_MACHINE = "x86_64"
EXPECTED_PYTHON_SHA256 = "ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866"
MAX_PYTHON_EXECUTABLE_BYTES = 128 * 1024 * 1024
MAX_PROBE_BYTES = 16 * 1024
MAX_PROBE_STDERR_BYTES = 4 * 1024
PROBE_TIMEOUT_SECONDS = 10
PROBE_CLEANUP_TIMEOUT_SECONDS = 1
SOURCE_ROOT = Path(__file__).resolve().parents[2]
UNSTABLE_PREFIXES = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))
PROBE_ENVIRONMENT = {"LC_ALL": "C", "PATH": "/usr/bin:/bin", "TZ": "UTC"}

CHECK_NAMES = (
    "external-no-follow-single-link-executable-held-and-hashed",
    "pinned-cpython-3-13-15-linux-x86-64-hash-matched-before-probe",
    "held-descriptor-path-ran-fixed-isolated-stdlib-probe",
    "tomllib-and-required-stdlib-imports-available-under-isolation",
    "interpreter-leaf-metadata-did-not-drift-during-probe",
    "preflight-does-not-install-or-materialize",
)

NOT_ESTABLISHED = {
    "interpreter_runtime_tree_immutability": "not-established",
    "runtime_tree_trust_before_probe_execution": "not-established",
    "post_hash_executable_content_immutability": "not-established",
    "same_uid_writer_exclusion": "not-established",
    "candidate_runtime_sandboxing": "not-established",
    "host_resource_or_hashing_time_isolation": "not-established",
    "toolchain_download_or_installation": "not-established",
    "same_fd_materializer_handoff": "not-established",
    "reconstructed_runtime_materialization": "not-established",
    "runtime_assembly_capture": "not-established",
    "docker_execution": "not-established",
    "service_execution": "not-established",
    "gpu_execution": "not-established",
    "evidence_creation": "not-established",
    "freeze": "not-established",
    "qualification": QUALIFICATION_STATUS,
}

# The payload is intentionally complete and constant. It has no
# caller-supplied script/module/environment/working-directory/network input.
# A mutable candidate runtime can still alter its own startup/import behavior,
# which is why the controller is a readiness diagnostic rather than a sandbox.
PROBE_PROGRAM = (
    "import bz2,hashlib,json,lzma,platform,sqlite3,sys,tarfile,tomllib;"
    "sys.stdout.write(json.dumps({'dont_write_bytecode':bool(sys.flags.dont_write_bytecode),"
    "'ignore_environment':bool(sys.flags.ignore_environment),"
    "'implementation':sys.implementation.name,'isolated':bool(sys.flags.isolated),"
    "'machine':platform.machine(),'no_site':bool(sys.flags.no_site),"
    "'no_user_site':bool(sys.flags.no_user_site),'platform':sys.platform,"
    "'tomllib':tomllib.__name__=='tomllib','version':platform.python_version()},"
    "sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False))"
)
PROBE_FIELDS = frozenset(
    {
        "dont_write_bytecode",
        "ignore_environment",
        "implementation",
        "isolated",
        "machine",
        "no_site",
        "no_user_site",
        "platform",
        "tomllib",
        "version",
    }
)


class ReconstructedRuntimePythonPrerequisiteError(ValueError):
    """A proposed Python interpreter cannot satisfy the fixed readiness pin."""


def _fail(code: str, message: str) -> NoReturn:
    error = ReconstructedRuntimePythonPrerequisiteError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _required_open_flag(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        _fail("missing-open-safety-flag", f"host does not expose required {name}")
    return value


def _safe_directory_flags() -> int:
    return (
        os.O_RDONLY
        | _required_open_flag("O_NOFOLLOW")
        | _required_open_flag("O_DIRECTORY")
        | _required_open_flag("O_CLOEXEC")
    )


def _safe_file_flags() -> int:
    return (
        os.O_RDONLY
        | _required_open_flag("O_NOFOLLOW")
        | _required_open_flag("O_CLOEXEC")
        | _required_open_flag("O_NONBLOCK")
    )


def _close_quietly(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _move_held_descriptor_above_standard_streams(descriptor: int) -> int:
    """Keep the executable FD distinct from child stdin/stdout/stderr setup."""

    if descriptor >= 3:
        return descriptor
    duplicate_flag = getattr(fcntl, "F_DUPFD_CLOEXEC", None)
    if type(duplicate_flag) is not int:
        _fail("missing-fcntl-safety-flag", "host does not expose F_DUPFD_CLOEXEC")
    try:
        duplicate = fcntl.fcntl(descriptor, duplicate_flag, 3)
    except OSError as error:
        _fail("python-descriptor-duplication-failed", f"cannot reserve held Python descriptor: {error}")
    if type(duplicate) is not int or duplicate < 3:
        _close_quietly(duplicate if type(duplicate) is int else None)
        _fail("python-descriptor-duplication-failed", "held Python descriptor did not move above standard streams")
    _close_quietly(descriptor)
    return duplicate


def _normalized_external_python_path(value: Path) -> Path:
    raw = os.fspath(value)
    if (
        type(raw) is not str
        or not raw
        or "\x00" in raw
        or "\n" in raw
        or "\r" in raw
        or not os.path.isabs(raw)
        or raw.startswith("//")
        or raw != os.path.normpath(raw)
    ):
        _fail("invalid-python-path", "--python must be one normalized absolute single-line path")
    path = Path(raw)
    if path == Path(os.path.sep) or not path.name or any(part in {"", ".", ".."} for part in path.parts):
        _fail("invalid-python-path", "--python must name one regular executable leaf")
    for unstable in UNSTABLE_PREFIXES:
        if path == unstable or unstable in path.parents:
            _fail("unstable-python-path", "--python must not live below a volatile temporary directory")
    try:
        path.relative_to(SOURCE_ROOT)
    except ValueError:
        return path
    _fail("python-path-inside-source-checkout", "--python must be external to the source checkout")


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_executable_metadata(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        _fail("unsafe-python-executable", "--python must be a regular executable file")
    if metadata.st_nlink != 1:
        _fail("nonunique-python-executable", "--python must have exactly one hard link")
    if metadata.st_uid not in {0, os.geteuid()}:
        _fail("unsafe-python-owner", "--python must be owned by root or the effective UID")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        _fail("unsafe-python-mode", "--python must not be group- or world-writable")
    if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID):
        _fail("unsafe-python-mode", "--python must not be setuid or setgid")
    if not metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
        _fail("nonexecutable-python", "--python must have at least one execute bit")
    if metadata.st_size < 1:
        _fail("invalid-python-size", "--python must not be empty")
    if metadata.st_size > MAX_PYTHON_EXECUTABLE_BYTES:
        _fail(
            "python-executable-too-large",
            "--python exceeds the bounded CPython executable size before hashing",
        )


def _open_held_python(path: Path) -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    directory_fd: int | None = None
    executable_fd: int | None = None
    try:
        components = path.parts
        directory_fd = os.open(os.path.sep, _safe_directory_flags())
        for component in components[1:-1]:
            try:
                child_fd = os.open(component, _safe_directory_flags(), dir_fd=directory_fd)
            except OSError as error:
                _fail("unsafe-python-directory", f"cannot open Python parent without links: {error}")
            _close_quietly(directory_fd)
            directory_fd = child_fd
        try:
            executable_fd = os.open(path.name, _safe_file_flags(), dir_fd=directory_fd)
        except OSError as error:
            _fail("unsafe-python-executable", f"cannot open --python without following links: {error}")
        try:
            visible = os.lstat(path.name, dir_fd=directory_fd)
            held = os.fstat(executable_fd)
        except OSError as error:
            _fail("unsafe-python-executable", f"cannot inspect held --python: {error}")
        if _stable_identity(visible) != _stable_identity(held):
            _fail("raced-python-executable", "visible --python leaf differs from the held descriptor")
        _validate_executable_metadata(held)
        identity = _stable_identity(held)
        result = _move_held_descriptor_above_standard_streams(executable_fd)
        executable_fd = None
        return result, identity
    finally:
        _close_quietly(executable_fd)
        _close_quietly(directory_fd)


def _sha256_held_file(descriptor: int) -> str:
    digest = hashlib.sha256()
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        _fail("python-hash-failed", f"cannot hash held --python: {error}")
    return digest.hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate-probe-json-key", f"Python probe repeats key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    _fail("non-finite-probe-json", f"Python probe emitted non-finite JSON number {value!r}")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        _fail("unencodable-canonical-json", f"cannot encode canonical preflight JSON: {error}")


def _parse_probe(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_PROBE_BYTES:
        _fail("invalid-probe-byte-length", "Python probe output has an invalid byte length")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
        )
    except ReconstructedRuntimePythonPrerequisiteError:
        raise
    except UnicodeDecodeError as error:
        _fail("invalid-probe-json", f"Python probe output is not UTF-8: {error}")
    except json.JSONDecodeError as error:
        _fail("invalid-probe-json", f"Python probe output is not JSON: {error}")
    except RecursionError as error:
        _fail("probe-json-nesting-too-deep", f"Python probe JSON nesting is unsafe: {error}")
    if type(document) is not dict or set(document) != PROBE_FIELDS:
        _fail("invalid-probe-schema", "Python probe output fields differ from the fixed contract")
    if raw != canonical_json_bytes(document):
        _fail("noncanonical-probe-json", "Python probe output must use exact canonical JSON")
    expected_text = {
        "implementation": EXPECTED_IMPLEMENTATION,
        "version": EXPECTED_VERSION,
        "platform": EXPECTED_PLATFORM,
        "machine": EXPECTED_MACHINE,
    }
    for field, expected in expected_text.items():
        if document[field] != expected:
            _fail("python-probe-mismatch", f"Python probe {field} must be {expected!r}")
    for field in ("isolated", "no_site", "ignore_environment", "no_user_site", "dont_write_bytecode", "tomllib"):
        if document[field] is not True:
            _fail("python-probe-mismatch", f"Python probe {field} must be true")
    return document


def _probe_argv(descriptor: int) -> list[str]:
    if type(descriptor) is not int or descriptor < 0:
        _fail("invalid-python-descriptor", "held Python descriptor must be a non-negative integer")
    return [f"/proc/self/fd/{descriptor}", "-I", "-S", "-E", "-B", "-c", PROBE_PROGRAM]


def _launch_held_probe(descriptor: int) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            _probe_argv(descriptor),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            cwd=os.path.sep,
            env=dict(PROBE_ENVIRONMENT),
            pass_fds=(descriptor,),
            start_new_session=True,
        )
    except OSError as error:
        _fail("python-probe-launch-failed", f"cannot execute held Python probe: {error}")


def _close_stream_quietly(stream: Any) -> None:
    if stream is not None:
        try:
            stream.close()
        except OSError:
            pass


def _terminate_probe_group(process: subprocess.Popen[bytes]) -> None:
    # The leader can exit after forking a descendant that still holds either
    # pipe.  Kill the new process group even when ``poll()`` already reports a
    # leader return code: the group remains addressable until its descendants
    # have exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, OSError):
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
    try:
        process.wait(timeout=PROBE_CLEANUP_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass


def _collect_probe_output(process: subprocess.Popen[bytes]) -> tuple[int, bytes, bytes]:
    """Collect a trusted-runtime probe with strict stdout/stderr memory caps.

    The candidate runtime tree is not a sandboxed input. Bounding both pipes
    nevertheless prevents a changed runtime from retaining unbounded bytes in
    the controller before the canonical parser can reject it. ``start_new_session``
    lets timeout/cap failure kill the probe's entire new process group.
    """

    streams = {"stdout": process.stdout, "stderr": process.stderr}
    limits = {"stdout": MAX_PROBE_BYTES, "stderr": MAX_PROBE_STDERR_BYTES}
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + PROBE_TIMEOUT_SECONDS
    selector: selectors.BaseSelector | None = None
    completed = False
    try:
        if process.stdout is None or process.stderr is None:
            _fail("python-probe-pipes-unavailable", "held Python probe did not expose both byte pipes")
        selector = selectors.DefaultSelector()
        for label, stream in streams.items():
            try:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, label)
            except (OSError, ValueError) as error:
                _fail("python-probe-pipes-unavailable", f"cannot safely read probe {label}: {error}")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _fail("python-probe-timeout", f"held Python probe exceeded {PROBE_TIMEOUT_SECONDS}s")
            events = selector.select(remaining)
            if not events:
                continue
            for key, _mask in events:
                stream = key.fileobj
                label = key.data
                remaining_capacity = limits[label] - len(outputs[label])
                try:
                    # Read at most one byte beyond the retained-capacity so a
                    # cap violation is detectable without retaining an
                    # arbitrary read chunk in the controller.
                    chunk = os.read(stream.fileno(), min(8192, remaining_capacity + 1))
                except BlockingIOError:
                    continue
                except OSError as error:
                    _fail("python-probe-read-failed", f"cannot read probe {label}: {error}")
                if not chunk:
                    selector.unregister(stream)
                    _close_stream_quietly(stream)
                    continue
                if len(chunk) > remaining_capacity:
                    _fail("python-probe-output-too-large", f"held Python probe {label} exceeds its byte limit")
                outputs[label].extend(chunk)
        try:
            returncode = process.wait(timeout=PROBE_CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _fail("python-probe-timeout", "held Python probe did not exit after its pipes closed")
        completed = True
        return returncode, bytes(outputs["stdout"]), bytes(outputs["stderr"])
    finally:
        # This includes interruption of the controller itself.  The child is
        # in a separate session, so Ctrl-C need not reach it; do not leave an
        # external interpreter behind on any non-successful collection path.
        if not completed:
            _terminate_probe_group(process)
        if selector is not None:
            selector.close()
        for stream in streams.values():
            _close_stream_quietly(stream)


def _run_held_probe(descriptor: int) -> dict[str, Any]:
    process = _launch_held_probe(descriptor)
    returncode, stdout, stderr = _collect_probe_output(process)
    if returncode != 0:
        _fail("python-probe-failed", f"held Python probe exited {returncode}")
    if stderr:
        _fail("unexpected-probe-stderr", "held Python probe wrote unexpected stderr")
    return _parse_probe(stdout)


def check_reconstructed_runtime_python_prerequisite(python: Path) -> dict[str, Any]:
    """Return a transient readiness result for one explicit external Python.

    No value returned here is a materialization result or a handoff capability.
    A later materializer must be invoked independently with the same absolute
    interpreter path after any required runtime-tree verification.
    """

    path = _normalized_external_python_path(python)
    descriptor: int | None = None
    try:
        descriptor, initial_identity = _open_held_python(path)
        sha256 = _sha256_held_file(descriptor)
        if sha256 != EXPECTED_PYTHON_SHA256:
            _fail("python-sha256-mismatch", "held --python SHA-256 differs from the pinned CPython 3.13.15 binary")
        probe = _run_held_probe(descriptor)
        try:
            terminal_identity = _stable_identity(os.fstat(descriptor))
        except OSError as error:
            _fail("unsafe-python-executable", f"cannot re-inspect held --python: {error}")
        if terminal_identity != initial_identity:
            _fail("python-executable-drift", "held --python changed during the isolated probe")
        return {
            "schema_version": PREREQUISITE_VERSION,
            "scope": SCOPE,
            "status": STATUS,
            "authority": AUTHORITY,
            "qualification_status": QUALIFICATION_STATUS,
            "python": {
                "path": os.fspath(path),
                "sha256": sha256,
                "byte_length": initial_identity[4],
                "implementation": probe["implementation"],
                "version": probe["version"],
                "platform": probe["platform"],
                "machine": probe["machine"],
            },
            "checks": [{"name": name, "satisfied": True} for name in CHECK_NAMES],
            "not_established": dict(NOT_ESTABLISHED),
            "reason_codes": [],
        }
    finally:
        _close_quietly(descriptor)


def _require_controller_isolation() -> None:
    if not (
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.ignore_environment
        and sys.flags.dont_write_bytecode
    ):
        _fail(
            "unsafe-controller-python",
            "invoke this prerequisite checker with python3 -I -S -E -B",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check one external pinned Python before reconstructed-runtime materialization."
    )
    parser.add_argument("--python", type=Path, required=True, help="absolute external CPython 3.13.15 path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        _require_controller_isolation()
        result = check_reconstructed_runtime_python_prerequisite(arguments.python)
    except ReconstructedRuntimePythonPrerequisiteError as error:
        reason = getattr(error, "reason_code", "unsafe-python-prerequisite")
        print(f"reconstructed runtime Python prerequisite: {reason}: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
