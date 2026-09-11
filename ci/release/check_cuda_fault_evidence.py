#!/usr/bin/env python3
"""Validate and package raw CUDA fault-injection evidence without running CUDA.

The GPU runner deliberately emits plain logs because its container contains no
Python runtime.  This standard-library-only host checker treats that directory
as hostile input, re-hashes its closed inventory, validates the subprocess and
ELF evidence, and only then creates the release-candidate attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import stat
import struct
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

from release_common import (
    ReleaseContractError,
    canonical_json_bytes,
    inspect_elf_dynamic,
    validate_binary,
)
from verify_release_bundle import verify_bundle


ATTESTATION_VERSION = "riley.release-gate-attestation.v1"
GATE = "cuda-fault-injection"
CHECK_IDS = (
    "test_inventory_exact",
    "create_rollback_ambiguity",
    "explicit_close_ambiguity",
    "confirmed_completion_deferred_error",
    "unconfirmed_completion_retained",
    "subprocess_isolation",
    "production_fault_symbols_absent",
)
FAULT_TESTS = {
    "memory_fault_cases_are_subprocess_isolated",
    "memory_fault_subprocess",
}
HOST_RUNTIME_TESTS = {
    "async_fill_is_correct_after_sync",
    "command_batch_proxy_is_one_shot_and_drop_restores_stream_use",
    "device_metadata_is_reported",
    "events_report_positive_elapsed_time",
    "invalid_device_is_rejected",
    "invalid_launch_reports_launch_stage",
    "repeated_create_drop_has_no_resource_leak",
    "two_stream_event_ordering_is_explicit",
}
MEMORY_TESTS = {
    "allocation_accounting_returns_to_zero",
    "copy_ranges_and_context_ownership_are_validated",
    "pinned_host_device_round_trip_is_exact",
    "two_stream_copy_handoff_prevents_early_reuse",
    "zero_byte_allocations_and_copies_are_logical_noops",
}
FAULT_CASES = (
    "create-rollback-ambiguous",
    "explicit-close-ambiguous",
    "deferred-submission-error",
    "completion-restore-ambiguous",
)

BASE_EVIDENCE_FILES = {
    "SHA256SUMS",
    "cuda-driver-libraries.txt",
    "environment.txt",
    "host-runtime-ldd.txt",
    "host-runtime-nm.txt",
    "host-runtime-readelf.txt",
    "host-runtime-test-binary",
    "host-runtime-test-binary.sha256",
    "host-runtime-test-list.txt",
    "host-runtime-tests.log",
    "memory-fault-test-binary.sha256",
    "memory-fault-test-binary",
    "memory-fault-ldd.txt",
    "memory-fault-nm.txt",
    "memory-fault-readelf.txt",
    "memory-fault-test-list.txt",
    "memory-fault-tests.log",
    "memory-ldd.txt",
    "memory-nm.txt",
    "memory-readelf.txt",
    "memory-test-binary",
    "memory-test-binary.sha256",
    "memory-test-list.txt",
    "memory-tests.log",
    "nvidia-smi-device-metadata.csv",
    "nvidia-smi-list.txt",
    "release-binary.sha256",
    "release-ldd.txt",
    "release-nm.txt",
    "release-readelf.txt",
}
SANITIZER_FILES = {
    "compute-sanitizer-memcheck.log",
    "compute-sanitizer-memory-memcheck.log",
}

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._+-]*)$")
TEST_BINARY_RE = re.compile(
    r"^([0-9a-f]{64})  "
    r"(target/debug/deps/(?:host_runtime_gpu|memory_gpu|memory_fault_injection_gpu)-[0-9a-f]{16})$"
)
NVIDIA_LIST_RE = re.compile(
    r"^GPU (?P<index>[0-9]+): (?P<name>[^\r\n]+) "
    r"\(UUID: (?P<uuid>GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})\)$"
)
NVIDIA_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
DRIVER_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?$")
COMPUTE_CAPABILITY_RE = re.compile(r"^[0-9]+\.[0-9]+$")
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4090"
EXPECTED_COMPUTE_CAPABILITY = "8.9"
FAULT_PREFIX = "riley_cuda_test_memory_fault_"
FORBIDDEN_DEPENDENCY_RE = re.compile(
    r"(?:libpython|pytorch|python|torch|transformers|triton|pickle)", re.IGNORECASE
)
SUMMARY = (
    "test result: ok. 1 passed; 0 failed; 0 ignored; "
    "0 measured; 1 filtered out;"
)
MARKER_PREFIX = "riley-cuda-memory-fault-case"
MARKER_RE = re.compile(
    rf"{MARKER_PREFIX} case=(?P<case>[a-z-]+) event=(?P<event>spawn|start|passed|joined) "
    r"(?:(?:parent_pid=(?P<parent>[1-9][0-9]*) )?)"
    r"child_pid=(?P<child>[1-9][0-9]*)"
    r"(?: exit_code=(?P<exit>-?[0-9]+))?(?=\r?$)",
    re.MULTILINE,
)

MAX_EVIDENCE_FILE_BYTES = 128 * 1024 * 1024
MAX_EVIDENCE_TOTAL_BYTES = 512 * 1024 * 1024
MAX_RAW_EVIDENCE_ARCHIVE_BYTES = MAX_EVIDENCE_TOTAL_BYTES + 4 * 1024 * 1024
MAX_SOURCE_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
MAX_RELEASE_BINARY_BYTES = 2 * 1024 * 1024 * 1024


class CudaFaultEvidenceError(ValueError):
    """Raw CUDA fault evidence is malformed, incomplete, or inconsistently bound."""


def _fail(path: str, message: str) -> NoReturn:
    raise CudaFaultEvidenceError(f"{path}: {message}")


def _regular_file(path: Path, label: str, maximum: int) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        _fail(label, f"cannot inspect {path}: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(label, "must be a regular file, not a link or device")
    if metadata.st_size > maximum:
        _fail(label, f"exceeds the {maximum}-byte bound")
    return path


def _sha256_file(path: Path, label: str, maximum: int) -> str:
    path = _regular_file(path, label, maximum)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _fail(label, f"cannot hash {path}: {error}")
    return digest.hexdigest()


def _text(contents: bytes, label: str, *, ascii_only: bool = False) -> str:
    try:
        return contents.decode("ascii" if ascii_only else "utf-8")
    except UnicodeDecodeError as error:
        _fail(label, f"is not valid {'ASCII' if ascii_only else 'UTF-8'}: {error}")


def _parse_environment(contents: bytes) -> dict[str, str]:
    text = _text(contents, "environment.txt")
    if "\0" in text or not text.endswith("\n"):
        _fail("environment.txt", "must be newline-terminated UTF-8 text without NUL bytes")
    required = {
        "source_revision",
        "source_archive_command",
        "source_archive_sha256",
        "gpu_image_id",
        "cuda_visible_devices",
        "nvidia_visible_devices",
        "leak_iterations",
        "compute_sanitizer",
    }
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key not in required:
            continue
        if key in result:
            _fail("environment.txt", f"repeats binding {key!r}")
        result[key] = value
    if set(result) != required:
        _fail(
            "environment.txt",
            f"binding set mismatch; missing={sorted(required - set(result))}",
        )
    if GIT_RE.fullmatch(result["source_revision"]) is None:
        _fail("environment.txt.source_revision", "must be a lowercase 40-character Git SHA")
    if result["source_archive_command"] != "git archive --format=tar HEAD":
        _fail("environment.txt.source_archive_command", "is not the reviewed archive command")
    if SHA_RE.fullmatch(result["source_archive_sha256"]) is None:
        _fail("environment.txt.source_archive_sha256", "must be a lowercase SHA-256")
    if IMAGE_RE.fullmatch(result["gpu_image_id"]) is None:
        _fail("environment.txt.gpu_image_id", "must be an immutable sha256 image ID")
    if result["compute_sanitizer"] not in {"0", "1"}:
        _fail("environment.txt.compute_sanitizer", "must be 0 or 1")
    if not result["leak_iterations"].isdigit() or not 32 <= int(result["leak_iterations"]) <= 4096:
        _fail("environment.txt.leak_iterations", "must be an integer from 32 through 4096")
    for key in ("cuda_visible_devices", "nvidia_visible_devices"):
        value = result[key]
        if (
            not value
            or value.casefold() in {"none", "void"}
            or any(character.isspace() for character in value)
            or "," in value
        ):
            _fail(f"environment.txt.{key}", "must select exactly one visible GPU or all")
        if value != "all" and value != "0" and NVIDIA_UUID_RE.fullmatch(value) is None:
            _fail(
                f"environment.txt.{key}",
                "must be all, device index 0, or a concrete NVIDIA GPU UUID",
            )
    required_runtime_patterns = {
        "Linux x86_64 host": r"(?m)^.*Linux.*x86_64.*$",
        "pinned Rust toolchain": r"(?m)^rustc 1\.85\.0(?: |$)",
        "pinned Cargo toolchain": r"(?m)^cargo 1\.85\.0(?: |$)",
        "CUDA 12.8 compiler": r"(?m)^Cuda compilation tools, release 12\.8, V12\.8\.[0-9]+$",
        "release executable version": (
            r"(?m)^riley 0\.1\.0 "
            r"\(server=true, cuda=true, cuda_abi=1\)$"
        ),
    }
    missing_runtime = [
        label for label, pattern in required_runtime_patterns.items()
        if re.search(pattern, text) is None
    ]
    if missing_runtime:
        _fail(
            "environment.txt",
            "does not prove the reviewed CUDA build/runtime environment: "
            + ", ".join(missing_runtime),
        )
    return result


def _parse_checksums(contents: bytes, expected: set[str]) -> dict[str, str]:
    text = _text(contents, "SHA256SUMS", ascii_only=True)
    if not text.endswith("\n"):
        _fail("SHA256SUMS", "must end with a newline")
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = CHECKSUM_RE.fullmatch(line)
        if match is None:
            _fail("SHA256SUMS", f"invalid line {line!r}")
        digest, name = match.groups()
        if name in result:
            _fail("SHA256SUMS", f"duplicate path {name!r}")
        result[name] = digest
    if set(result) != expected:
        _fail(
            "SHA256SUMS",
            f"path set mismatch; missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}",
        )
    if list(result) != sorted(result):
        _fail("SHA256SUMS", "paths must be bytewise sorted")
    return result


def _single_test_binary_checksum(
    contents: bytes,
    label: str,
    expected_stem: str,
) -> tuple[str, str]:
    text = _text(contents, label, ascii_only=True)
    if not text.endswith("\n") or len(text.splitlines()) != 1:
        _fail(label, "must contain exactly one newline-terminated sha256sum record")
    match = TEST_BINARY_RE.fullmatch(text.rstrip("\n"))
    if match is None:
        _fail(label, "has an invalid test-binary checksum record")
    digest, path = match.groups()
    if not PurePosixPath(path).name.startswith(f"{expected_stem}-"):
        _fail(label, f"does not identify the {expected_stem} executable")
    return digest, path


def _validate_test_inventory(
    contents: bytes,
    label: str,
    expected: set[str],
) -> None:
    text = _text(contents, label)
    if not text.endswith("\n"):
        _fail(label, "must be newline terminated")
    observed = [
        line.removesuffix(": test")
        for line in text.splitlines()
        if line.endswith(": test")
    ]
    if len(observed) != len(set(observed)):
        _fail(label, "contains a duplicate test entry")
    if set(observed) != expected:
        _fail(
            label,
            f"reviewed test inventory mismatch; expected={sorted(expected)}, "
            f"observed={sorted(observed)}",
        )


def _validate_fault_inventory(contents: bytes) -> None:
    try:
        _validate_test_inventory(
            contents,
            "memory-fault-test-list.txt",
            FAULT_TESTS,
        )
    except CudaFaultEvidenceError as error:
        _fail(
            "memory-fault-test-list.txt",
            f"must list exactly the two reviewed tests ({error})",
        )


def _validate_test_execution_log(
    contents: bytes,
    label: str,
    expected: set[str],
) -> str:
    text = _text(contents, label)
    expected_count = len(expected)
    if "\0" in text or not text.endswith("\n"):
        _fail(label, "must be newline-terminated UTF-8 text without NUL bytes")
    if "FAILED" in text or "test result: FAILED" in text:
        _fail(label, "contains a failed test result")
    if len(re.findall(rf"(?m)^running {expected_count} tests\r?$", text)) != 1:
        _fail(label, f"must contain exactly one 'running {expected_count} tests' line")

    headings = list(re.finditer(r"(?m)^test ([a-z0-9_]+) \.\.\.", text))
    observed = [match.group(1) for match in headings]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        _fail(
            label,
            f"executed test inventory mismatch; expected={sorted(expected)}, "
            f"observed={sorted(observed)}",
        )
    summary_re = re.compile(
        rf"(?m)^test result: ok\. {expected_count} passed; 0 failed; 0 ignored; "
        r"0 measured; 0 filtered out; finished in [0-9]+(?:\.[0-9]+)?s\r?$"
    )
    summaries = list(summary_re.finditer(text))
    if len(summaries) != 1:
        _fail(label, "must contain exactly one complete passing libtest summary")

    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else summaries[0].start()
        segment = text[heading.start():end]
        if re.search(r"(?:\.\.\. ok(?:\r?$)|^ok\r?$)", segment, re.MULTILINE) is None:
            _fail(label, f"does not record an ok outcome for {heading.group(1)}")
    return text


def _validate_linux_x86_64_test_elf(
    binary: bytes,
    label: str,
    *,
    required_strings: set[str],
    require_fault_symbols: bool = False,
) -> list[str]:
    if len(binary) < 64 or binary[:4] != b"\x7fELF":
        _fail(label, "is not an ELF file")
    if binary[4:7] != bytes((2, 1, 1)) or binary[7] not in (0, 3):
        _fail(label, "must be a 64-bit little-endian Linux/System-V ELF")
    try:
        header = struct.unpack_from("<HHIQQQIHHHHHH", binary, 16)
    except struct.error as error:
        _fail(label, f"has a truncated ELF header: {error}")
    elf_type, machine, version = header[:3]
    entry, phoff, phentsize, phnum = header[3], header[4], header[8], header[9]
    if elf_type not in (2, 3) or machine != 62 or version != 1 or entry == 0:
        _fail(label, "must be an executable Linux x86-64 ET_EXEC/ET_DYN ELF")
    if phentsize != 56 or phnum == 0 or phoff + phentsize * phnum > len(binary):
        _fail(label, "has an invalid ELF program-header table")
    executable_load = False
    for index in range(phnum):
        try:
            segment = struct.unpack_from("<IIQQQQQQ", binary, phoff + index * phentsize)
        except struct.error as error:
            _fail(label, f"has a truncated program header: {error}")
        segment_type, flags, offset, file_size = segment[0], segment[1], segment[2], segment[5]
        if offset + file_size > len(binary):
            _fail(label, "contains an out-of-range ELF segment")
        if segment_type == 1 and flags & 1 and file_size > 0:
            executable_load = True
    if not executable_load:
        _fail(label, "does not contain an executable PT_LOAD segment")
    try:
        dependencies, dynamic_paths = inspect_elf_dynamic(binary)
    except ReleaseContractError as error:
        _fail(label, f"ELF dynamic inspection failed: {error}")
    if dynamic_paths:
        _fail(label, "contains DT_RPATH or DT_RUNPATH")
    if "libcuda.so.1" not in dependencies or not any(
        dependency.startswith("libcudart.so") for dependency in dependencies
    ):
        _fail(label, "does not directly link the CUDA driver and runtime")
    if any(FORBIDDEN_DEPENDENCY_RE.search(dependency) for dependency in dependencies):
        _fail(label, "contains a forbidden dynamic dependency")
    missing_strings = sorted(
        value for value in required_strings if value.encode("ascii") not in binary
    )
    if missing_strings:
        _fail(label, f"omits reviewed test/marker strings: {missing_strings}")
    has_fault_symbol = FAULT_PREFIX.encode("ascii") in binary
    if require_fault_symbols and not has_fault_symbol:
        _fail(label, "does not contain the test-only CUDA fault-injection symbol prefix")
    if not require_fault_symbols and has_fault_symbol:
        _fail(label, "unexpectedly contains the CUDA fault-injection symbol prefix")
    return dependencies


def _parse_nvidia_evidence(files: dict[str, bytes], environment: dict[str, str]) -> dict[str, Any]:
    list_text = _text(files["nvidia-smi-list.txt"], "nvidia-smi-list.txt", ascii_only=True)
    csv_text = _text(
        files["nvidia-smi-device-metadata.csv"],
        "nvidia-smi-device-metadata.csv",
        ascii_only=True,
    )
    if not list_text.endswith("\n") or not csv_text.endswith("\n"):
        _fail("nvidia-smi evidence", "both files must be newline terminated")
    list_lines = list_text.splitlines()
    csv_lines = csv_text.splitlines()
    if len(list_lines) != 1 or len(csv_lines) != 1:
        _fail("nvidia-smi evidence", "must describe exactly one visible GPU")
    list_match = NVIDIA_LIST_RE.fullmatch(list_lines[0])
    if list_match is None:
        _fail("nvidia-smi-list.txt", "does not contain the canonical one-GPU inventory")
    fields = [field.strip() for field in csv_lines[0].split(",")]
    if len(fields) != 6:
        _fail("nvidia-smi-device-metadata.csv", "must contain exactly six query fields")
    index, uuid, name, capability, memory_mib_text, driver = fields
    if not index.isdigit() or NVIDIA_UUID_RE.fullmatch(uuid) is None:
        _fail("nvidia-smi-device-metadata.csv", "has an invalid index or UUID")
    if COMPUTE_CAPABILITY_RE.fullmatch(capability) is None:
        _fail("nvidia-smi-device-metadata.csv", "has an invalid compute capability")
    if not memory_mib_text.isdigit() or int(memory_mib_text) < 1024:
        _fail("nvidia-smi-device-metadata.csv", "has an invalid total-memory value")
    if DRIVER_VERSION_RE.fullmatch(driver) is None:
        _fail("nvidia-smi-device-metadata.csv", "has an invalid driver version")
    if (index, uuid, name) != (
        list_match.group("index"),
        list_match.group("uuid"),
        list_match.group("name"),
    ):
        _fail("nvidia-smi evidence", "list and query rows identify different devices")
    if name != EXPECTED_GPU_NAME or capability != EXPECTED_COMPUTE_CAPABILITY:
        _fail(
            "nvidia-smi-device-metadata.csv",
            "does not identify the reviewed RTX 4090/sm89 release GPU",
        )
    for key in ("cuda_visible_devices", "nvidia_visible_devices"):
        selected = environment[key]
        if selected not in {"all", "0", uuid}:
            _fail(f"environment.txt.{key}", "does not select the recorded GPU")
    return {
        "index": int(index),
        "uuid": uuid,
        "name": name,
        "compute_capability": capability,
        "memory_mib": int(memory_mib_text),
        "driver_version": driver,
    }


def _validate_driver_library_inventory(contents: bytes) -> None:
    text = _text(contents, "cuda-driver-libraries.txt")
    if not text.endswith("\n") or "/stubs/" in text:
        _fail(
            "cuda-driver-libraries.txt",
            "must be newline terminated and must not resolve CUDA driver stubs",
        )
    for library in ("libcuda.so.1", "libcudart.so"):
        if library not in text:
            _fail("cuda-driver-libraries.txt", f"does not resolve {library}")


def _validate_test_elf_logs(
    files: dict[str, bytes],
    *,
    prefix: str,
    artifact_path: str,
    binary_sha256: str,
    dependencies: list[str],
    require_fault_symbols: bool = False,
) -> None:
    headers = f"artifact={artifact_path}\nsha256={binary_sha256}\n"
    labels = {
        "ldd": f"{prefix}-ldd.txt",
        "readelf": f"{prefix}-readelf.txt",
        "nm": f"{prefix}-nm.txt",
    }
    values: dict[str, str] = {}
    for kind, label in labels.items():
        value = _text(files[label], label)
        if not value.startswith(headers):
            _fail(label, "does not bind the exact inspected ELF path and SHA-256")
        body = value[len(headers):]
        if not body.strip():
            _fail(label, "contains no inspection output")
        values[kind] = body

    ldd = values["ldd"]
    readelf = values["readelf"]
    nm = values["nm"]
    if re.search(r"=>\s+not found", ldd):
        _fail(labels["ldd"], "contains an unresolved dependency")
    if "/stubs/" in ldd or "/stubs/" in readelf:
        _fail(f"{prefix} ELF evidence", "resolves or embeds CUDA driver stubs")
    needed = re.findall(r"NEEDED[^\r\n]*Shared library: \[([^\]]+)\]", readelf)
    if needed != list(dict.fromkeys(needed)) or set(needed) != set(dependencies):
        _fail(
            labels["readelf"],
            "DT_NEEDED inventory differs from the preserved ELF bytes",
        )
    if re.search(r"\b(?:RPATH|RUNPATH)\b", readelf):
        _fail(labels["readelf"], "contains an unreviewed runtime search path")
    for library in ("libcudart.so", "libcuda.so.1"):
        if library not in ldd:
            _fail(labels["ldd"], f"does not resolve {library}")
    if FORBIDDEN_DEPENDENCY_RE.search(ldd) or FORBIDDEN_DEPENDENCY_RE.search(readelf):
        _fail(f"{prefix} ELF evidence", "contains a forbidden runtime dependency")
    has_fault_symbol = FAULT_PREFIX in nm
    if require_fault_symbols and not has_fault_symbol:
        _fail(labels["nm"], "does not show the test-only CUDA fault symbols")
    if not require_fault_symbols and has_fault_symbol:
        _fail(labels["nm"], "unexpectedly shows a CUDA fault-injection symbol")




def _validate_fault_log(contents: bytes) -> None:
    text = _text(contents, "memory-fault-tests.log")
    if "FAILED" in text or len(re.findall(r"(?m)^running 1 test\r?$", text)) != 5:
        _fail(
            "memory-fault-tests.log",
            "must contain five isolated one-test executions and no failure",
        )
    matches = list(MARKER_RE.finditer(text))
    if text.count(MARKER_PREFIX) != 16 or len(matches) != 16:
        _fail("memory-fault-tests.log", "must contain exactly four markers for each fault case")

    by_case: dict[str, dict[str, re.Match[str]]] = {
        case: {} for case in FAULT_CASES
    }
    for match in matches:
        case = match.group("case")
        event = match.group("event")
        if case not in by_case:
            _fail("memory-fault-tests.log", f"contains unknown fault case {case!r}")
        if event in by_case[case]:
            _fail("memory-fault-tests.log", f"duplicates {case!r} {event!r} marker")
        parent = match.group("parent")
        exit_code = match.group("exit")
        if event in {"spawn", "joined"}:
            if parent is None:
                _fail("memory-fault-tests.log", f"{case} {event} omits parent PID")
        elif parent is not None:
            _fail("memory-fault-tests.log", f"{case} {event} unexpectedly records parent PID")
        if (event == "joined") != (exit_code is not None):
            _fail("memory-fault-tests.log", f"{case} has an invalid exit-code marker")
        if event == "joined" and exit_code != "0":
            _fail("memory-fault-tests.log", f"{case} child did not exit zero")
        by_case[case][event] = match

    parent_pids: set[str] = set()
    child_pids: set[str] = set()
    previous_join_position = -1
    for case in FAULT_CASES:
        events = by_case[case]
        if set(events) != {"spawn", "start", "passed", "joined"}:
            _fail("memory-fault-tests.log", f"{case} marker set is incomplete")
        positions = {event: match.start() for event, match in events.items()}
        # The child can print `start` before the parent gets scheduled to print
        # `spawn` after Command::spawn returns.  Both streams must still finish
        # before join, and no next case may begin before the preceding join.
        if not (
            positions["start"] < positions["passed"] < positions["joined"]
            and positions["spawn"] < positions["joined"]
            and min(positions.values()) > previous_join_position
        ):
            _fail("memory-fault-tests.log", f"{case} markers are not sequentially isolated")
        previous_join_position = positions["joined"]
        case_children = {match.group("child") for match in events.values()}
        if len(case_children) != 1:
            _fail("memory-fault-tests.log", f"{case} markers disagree on child PID")
        child_pid = case_children.pop()
        if child_pid in child_pids:
            _fail("memory-fault-tests.log", "fault cases reused a child process")
        child_pids.add(child_pid)
        case_parents = {
            events[event].group("parent") for event in ("spawn", "joined")
        }
        if len(case_parents) != 1:
            _fail("memory-fault-tests.log", f"{case} markers disagree on parent PID")
        parent_pid = case_parents.pop()
        if parent_pid == child_pid:
            _fail("memory-fault-tests.log", f"{case} ran in the parent process")
        parent_pids.add(parent_pid)
    if len(parent_pids) != 1:
        _fail("memory-fault-tests.log", "fault cases did not share one parent harness")

    if text.count(SUMMARY) != 5:
        _fail(
            "memory-fault-tests.log",
            "must contain four child and one parent passing test summaries",
        )
    child_runs = len(
        re.findall(r"(?m)^test memory_fault_subprocess \.\.\.", text)
    )
    parent_runs = len(
        re.findall(
            r"(?m)^test memory_fault_cases_are_subprocess_isolated \.\.\.",
            text,
        )
    )
    successful_outcomes = len(
        re.findall(r"(?m)(?:\.\.\. ok\r?$|^ok\r?$)", text)
    )
    if child_runs != 4 or parent_runs != 1 or successful_outcomes != 5:
        _fail("memory-fault-tests.log", "does not prove four child passes and one parent pass")


def _validate_host_runtime_log(
    contents: bytes,
    environment: dict[str, str],
    gpu: dict[str, Any],
) -> None:
    text = _validate_test_execution_log(
        contents,
        "host-runtime-tests.log",
        HOST_RUNTIME_TESTS,
    )
    metadata_re = re.compile(
        r"(?m)riley-cuda-device-metadata "
        r"ordinal=(?P<ordinal>[0-9]+) "
        r"name=(?P<name>.+?) "
        r"compute_capability=(?P<capability>[0-9]+\.[0-9]+) "
        r"total_memory_bytes=(?P<memory>[1-9][0-9]*) "
        r"multiprocessor_count=(?P<multiprocessors>[1-9][0-9]*) "
        r"driver_version=(?P<driver>[1-9][0-9]*) "
        r"runtime_version=(?P<runtime>[1-9][0-9]*)\r?$"
    )
    metadata = list(metadata_re.finditer(text))
    if len(metadata) != 1:
        _fail("host-runtime-tests.log", "must contain one complete CUDA device marker")
    marker = metadata[0]
    if (
        int(marker.group("ordinal")) != gpu["index"]
        or marker.group("name") != gpu["name"]
        or marker.group("capability") != gpu["compute_capability"]
    ):
        _fail(
            "host-runtime-tests.log",
            "CUDA runtime marker identifies a different device than nvidia-smi",
        )
    runtime_memory = int(marker.group("memory"))
    reported_memory = int(gpu["memory_mib"]) * 1024 * 1024
    if not reported_memory - 2 * 1024 * 1024 * 1024 <= runtime_memory <= reported_memory:
        _fail(
            "host-runtime-tests.log",
            "CUDA runtime memory is inconsistent with nvidia-smi total memory",
        )

    leak_re = re.compile(
        r"(?m)riley-cuda-leak-smoke "
        r"iterations=(?P<iterations>[0-9]+) "
        r"before_free_bytes=(?P<before>[1-9][0-9]*) "
        r"after_free_bytes=(?P<after>[1-9][0-9]*)\r?$"
    )
    leaks = list(leak_re.finditer(text))
    if len(leaks) != 1:
        _fail("host-runtime-tests.log", "must contain one complete CUDA leak marker")
    leak = leaks[0]
    if leak.group("iterations") != environment["leak_iterations"]:
        _fail("host-runtime-tests.log", "leak marker iteration count differs from environment")
    before = int(leak.group("before"))
    after = int(leak.group("after"))
    if after + 64 * 1024 * 1024 < before:
        _fail("host-runtime-tests.log", "free GPU memory dropped beyond the reviewed tolerance")


def _validate_memory_log(contents: bytes, label: str = "memory-tests.log") -> None:
    text = _validate_test_execution_log(contents, label, MEMORY_TESTS)
    accounting = (
        "riley-cuda-memory-accounting device_live_bytes=0 "
        "device_live_allocations=0 pinned_host_live_bytes=0 "
        "pinned_host_live_allocations=0"
    )
    if len(re.findall(rf"(?m)^{re.escape(accounting)}\r?$", text)) != 1:
        _fail(label, "must contain exactly one all-zero CUDA allocation marker")


def _validate_release_logs(
    files: dict[str, bytes],
    binary_sha256: str,
    dependencies: list[str],
) -> None:
    _validate_test_elf_logs(
        files,
        prefix="release",
        artifact_path="target/release/riley",
        binary_sha256=binary_sha256,
        dependencies=dependencies,
    )


def _validate_evidence_files(
    files: dict[str, bytes],
    *,
    inventory_label: str,
) -> dict[str, str]:
    if "environment.txt" not in files:
        _fail(inventory_label, "closed inventory is missing environment.txt")
    environment = _parse_environment(files["environment.txt"])
    expected = set(BASE_EVIDENCE_FILES)
    if environment["compute_sanitizer"] == "1":
        expected.update(SANITIZER_FILES)

    observed = set(files)
    if observed != expected:
        _fail(
            inventory_label,
            f"closed inventory mismatch; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}",
        )
    total = sum(len(contents) for contents in files.values())
    oversized = sorted(
        name for name, contents in files.items()
        if len(contents) > MAX_EVIDENCE_FILE_BYTES
    )
    if oversized:
        _fail(inventory_label, f"members exceed the per-file bound: {oversized}")
    if total > MAX_EVIDENCE_TOTAL_BYTES:
        _fail(inventory_label, f"exceeds the {MAX_EVIDENCE_TOTAL_BYTES}-byte total bound")

    checksums = _parse_checksums(files["SHA256SUMS"], expected - {"SHA256SUMS"})
    for name, expected_digest in checksums.items():
        actual = hashlib.sha256(files[name]).hexdigest()
        if actual != expected_digest:
            _fail("SHA256SUMS", f"digest mismatch for {name}: {actual}")
    return environment


def _read_evidence_directory(root: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    try:
        metadata = root.lstat()
    except OSError as error:
        _fail("--evidence-dir", f"cannot inspect {root}: {error}")
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("--evidence-dir", "must be a real directory, not a link")

    environment_path = root / "environment.txt"
    environment_path = _regular_file(
        environment_path, "environment.txt", MAX_EVIDENCE_FILE_BYTES
    )
    try:
        environment_bytes = environment_path.read_bytes()
    except OSError as error:
        _fail("environment.txt", f"cannot read evidence: {error}")
    environment = _parse_environment(environment_bytes)
    expected = set(BASE_EVIDENCE_FILES)
    if environment["compute_sanitizer"] == "1":
        expected.update(SANITIZER_FILES)

    try:
        entries = list(root.iterdir())
    except OSError as error:
        _fail("--evidence-dir", f"cannot enumerate {root}: {error}")
    observed = {entry.name for entry in entries}
    if observed != expected or len(entries) != len(observed):
        _fail(
            "--evidence-dir",
            f"closed inventory mismatch; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}",
        )

    files: dict[str, bytes] = {}
    total = 0
    for name in sorted(expected):
        path = _regular_file(root / name, name, MAX_EVIDENCE_FILE_BYTES)
        try:
            contents = path.read_bytes()
        except OSError as error:
            _fail(name, f"cannot read evidence: {error}")
        total += len(contents)
        if total > MAX_EVIDENCE_TOTAL_BYTES:
            _fail("--evidence-dir", f"exceeds the {MAX_EVIDENCE_TOTAL_BYTES}-byte total bound")
        files[name] = contents

    return files, _validate_evidence_files(files, inventory_label="--evidence-dir")


def load_raw_evidence_archive(
    path: Path,
) -> tuple[dict[str, bytes], dict[str, str], str]:
    """Load and verify the canonical, closed CUDA fault raw-evidence tar."""

    label = "--raw-evidence"
    raw_path = _regular_file(path, label, MAX_RAW_EVIDENCE_ARCHIVE_BYTES)
    try:
        archive_bytes = raw_path.read_bytes()
    except OSError as error:
        _fail(label, f"cannot read {path}: {error}")
    if not archive_bytes:
        _fail(label, "must not be empty")

    files: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            if archive.pax_headers:
                _fail(label, "global PAX headers are forbidden")
            members = archive.getmembers()
            names = [member.name for member in members]
            if names != sorted(names):
                _fail(label, "members must be bytewise sorted")
            expected_offset = 0
            for member in members:
                name = member.name
                if (
                    name in files
                    or name not in BASE_EVIDENCE_FILES | SANITIZER_FILES
                    or "/" in name
                    or PurePosixPath(name).name != name
                ):
                    _fail(label, f"unexpected or duplicate member {name!r}")
                if not member.isreg():
                    _fail(label, f"member must be a regular file: {name}")
                if member.pax_headers:
                    _fail(label, f"member PAX headers are forbidden: {name}")
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.uname != "root"
                    or member.gname != "root"
                    or member.mode != 0o644
                    or member.mtime != 0
                    or member.linkname != ""
                    or member.devmajor != 0
                    or member.devminor != 0
                ):
                    _fail(label, f"non-canonical metadata for {name}")
                if member.size > MAX_EVIDENCE_FILE_BYTES:
                    _fail(label, f"member exceeds the per-file bound: {name}")
                if member.offset != expected_offset or member.offset_data != expected_offset + 512:
                    _fail(label, f"non-canonical header layout for {name}")

                expected_info = tarfile.TarInfo(name)
                expected_info.size = member.size
                expected_info.mode = 0o644
                expected_info.uid = 0
                expected_info.gid = 0
                expected_info.uname = "root"
                expected_info.gname = "root"
                expected_info.mtime = 0
                expected_header = expected_info.tobuf(format=tarfile.PAX_FORMAT)
                actual_header = archive_bytes[member.offset:member.offset_data]
                if actual_header != expected_header:
                    _fail(label, f"non-canonical tar header for {name}")

                data_end = member.offset_data + member.size
                padded_end = member.offset_data + ((member.size + 511) // 512) * 512
                if data_end > len(archive_bytes):
                    _fail(label, f"truncated member {name}")
                if any(archive_bytes[data_end:padded_end]):
                    _fail(label, f"non-zero data padding for {name}")
                files[name] = archive_bytes[member.offset_data:data_end]
                expected_offset = padded_end
    except CudaFaultEvidenceError:
        raise
    except (OSError, tarfile.TarError, UnicodeError, ValueError) as error:
        _fail(label, f"is not a readable canonical uncompressed tar: {error}")

    canonical_size = ((expected_offset + 1024 + 10239) // 10240) * 10240
    if len(archive_bytes) != canonical_size or any(archive_bytes[expected_offset:]):
        _fail(label, "has non-canonical end-of-archive padding")
    environment = _validate_evidence_files(files, inventory_label=label)
    return files, environment, hashlib.sha256(archive_bytes).hexdigest()


def _validate_source_archive(path: Path, revision: str) -> str:
    digest = _sha256_file(path, "--source-archive", MAX_SOURCE_ARCHIVE_BYTES)
    try:
        with tarfile.open(path, "r:") as archive:
            comment = archive.pax_headers.get("comment")
            if comment != revision:
                _fail(
                    "--source-archive",
                    "PAX git archive commit does not match --source-revision",
                )
    except (OSError, tarfile.TarError) as error:
        _fail("--source-archive", f"is not a readable uncompressed git archive: {error}")
    return digest


def _strict_json(contents: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in items:
            if key in value:
                _fail(label, f"contains duplicate JSON key {key!r}")
            value[key] = child
        return value

    def nonfinite(value: str) -> NoReturn:
        _fail(label, f"contains non-finite JSON number {value!r}")

    try:
        value = json.loads(contents, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(label, f"is not strict UTF-8 JSON: {error}")
    if not isinstance(value, dict):
        _fail(label, "must be a JSON object")
    return value


def _validate_release_bundle(
    bundle: Path,
    *,
    binary_sha256: str,
    source_revision: str,
) -> str:
    bundle_sha256 = _sha256_file(
        bundle, "--release-bundle", MAX_SOURCE_ARCHIVE_BYTES
    )
    try:
        verify_bundle(bundle)
        with tarfile.open(bundle, "r:gz") as archive:
            binaries = [
                member for member in archive.getmembers()
                if member.name.endswith("/bin/riley")
            ]
            manifests = [
                member for member in archive.getmembers()
                if member.name.endswith("/manifest/release.json")
            ]
            if len(binaries) != 1 or len(manifests) != 1:
                _fail("--release-bundle", "must contain one binary and one release manifest")
            binary_file = archive.extractfile(binaries[0])
            manifest_file = archive.extractfile(manifests[0])
            if binary_file is None or manifest_file is None:
                _fail("--release-bundle", "cannot read embedded release artifacts")
            embedded_binary_sha256 = hashlib.sha256(binary_file.read()).hexdigest()
            manifest = _strict_json(manifest_file.read(), "release manifest")
    except (OSError, tarfile.TarError, ReleaseContractError) as error:
        _fail("--release-bundle", f"verification failed: {error}")
    if embedded_binary_sha256 != binary_sha256:
        _fail("--release-bundle", "embedded binary differs from --release-binary")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("source_revision") != source_revision:
        _fail("--release-bundle", "embedded source revision differs from candidate")
    return bundle_sha256


def _image_digest(value: str, label: str) -> str:
    match = IMAGE_RE.fullmatch(value)
    if match is None:
        _fail(label, "must be an immutable sha256:<lowercase digest> image ID")
    return match.group(1)


def _validate_bound_evidence(
    files: dict[str, bytes],
    environment: dict[str, str],
    *,
    source_revision: str,
    source_archive: Path,
    build_image_id: str,
    release_binary: Path,
    release_bundle: Path,
    release_image_id: str,
) -> dict[str, Any]:
    """Bind already-validated raw files to the immutable candidate artifacts."""

    if GIT_RE.fullmatch(source_revision) is None:
        _fail("--source-revision", "must be a lowercase 40-character Git SHA")
    _image_digest(build_image_id, "--build-image-id")
    release_image_sha256 = _image_digest(release_image_id, "--release-image-id")
    source_archive_sha256 = _validate_source_archive(source_archive, source_revision)

    binary_path = _regular_file(
        release_binary, "--release-binary", MAX_RELEASE_BINARY_BYTES
    )
    try:
        binary = binary_path.read_bytes()
    except OSError as error:
        _fail("--release-binary", f"cannot read {release_binary}: {error}")
    if FAULT_PREFIX.encode("ascii") in binary:
        _fail("--release-binary", "contains a test-only CUDA fault-injection symbol")
    try:
        release_dependencies = validate_binary(binary)
    except ReleaseContractError as error:
        _fail("--release-binary", f"ELF validation failed: {error}")
    release_binary_sha256 = hashlib.sha256(binary).hexdigest()
    release_bundle_sha256 = _validate_release_bundle(
        release_bundle,
        binary_sha256=release_binary_sha256,
        source_revision=source_revision,
    )

    expected_environment = {
        "source_revision": source_revision,
        "source_archive_sha256": source_archive_sha256,
        "gpu_image_id": build_image_id,
    }
    for key, expected in expected_environment.items():
        if environment[key] != expected:
            _fail(f"environment.txt.{key}", "does not match the supplied immutable input")

    gpu = _parse_nvidia_evidence(files, environment)
    _validate_driver_library_inventory(files["cuda-driver-libraries.txt"])
    _validate_test_inventory(
        files["host-runtime-test-list.txt"],
        "host-runtime-test-list.txt",
        HOST_RUNTIME_TESTS,
    )
    _validate_test_inventory(
        files["memory-test-list.txt"],
        "memory-test-list.txt",
        MEMORY_TESTS,
    )
    _validate_fault_inventory(files["memory-fault-test-list.txt"])
    _validate_host_runtime_log(files["host-runtime-tests.log"], environment, gpu)
    _validate_memory_log(files["memory-tests.log"])
    _validate_fault_log(files["memory-fault-tests.log"])

    host_digest, host_path = _single_test_binary_checksum(
        files["host-runtime-test-binary.sha256"],
        "host-runtime-test-binary.sha256",
        "host_runtime_gpu",
    )
    memory_digest, memory_path = _single_test_binary_checksum(
        files["memory-test-binary.sha256"],
        "memory-test-binary.sha256",
        "memory_gpu",
    )
    fault_digest, fault_path = _single_test_binary_checksum(
        files["memory-fault-test-binary.sha256"],
        "memory-fault-test-binary.sha256",
        "memory_fault_injection_gpu",
    )
    binary_specs = (
        (
            "host-runtime-test-binary",
            host_digest,
            host_path,
            HOST_RUNTIME_TESTS | {
                "riley-cuda-device-metadata",
                "riley-cuda-leak-smoke",
            },
            "host-runtime",
            False,
            "host-runtime-test-list.txt",
            "host-runtime-tests.log",
        ),
        (
            "memory-test-binary",
            memory_digest,
            memory_path,
            MEMORY_TESTS | {"riley-cuda-memory-accounting"},
            "memory",
            False,
            "memory-test-list.txt",
            "memory-tests.log",
        ),
        (
            "memory-fault-test-binary",
            fault_digest,
            fault_path,
            FAULT_TESTS
            | set(FAULT_CASES)
            | {MARKER_PREFIX, "RILEY_CUDA_MEMORY_FAULT_CHILD"},
            "memory-fault",
            True,
            "memory-fault-test-list.txt",
            "memory-fault-tests.log",
        ),
    )
    for (
        evidence_name,
        receipt_digest,
        artifact_path,
        required_strings,
        log_prefix,
        requires_fault_symbols,
        list_name,
        execution_name,
    ) in binary_specs:
        test_binary = files[evidence_name]
        actual_digest = hashlib.sha256(test_binary).hexdigest()
        if actual_digest != receipt_digest:
            _fail(
                f"{evidence_name}.sha256",
                f"checksum receipt does not match preserved {evidence_name} bytes",
            )
        for log_name in (list_name, execution_name):
            log_text = _text(files[log_name], log_name)
            if log_text.count(f"({artifact_path})") != 1:
                _fail(
                    log_name,
                    "does not identify exactly once the test executable from its checksum receipt",
                )
        dependencies = _validate_linux_x86_64_test_elf(
            test_binary,
            evidence_name,
            required_strings=required_strings,
            require_fault_symbols=requires_fault_symbols,
        )
        _validate_test_elf_logs(
            files,
            prefix=log_prefix,
            artifact_path=artifact_path,
            binary_sha256=receipt_digest,
            dependencies=dependencies,
            require_fault_symbols=requires_fault_symbols,
        )

    release_checksum = _text(
        files["release-binary.sha256"], "release-binary.sha256", ascii_only=True
    )
    expected_release_checksum = f"{release_binary_sha256}  target/release/riley\n"
    if release_checksum != expected_release_checksum:
        _fail(
            "release-binary.sha256",
            "does not bind the inspected production release binary",
        )
    _validate_release_logs(files, release_binary_sha256, release_dependencies)

    if environment["compute_sanitizer"] == "1":
        for name in sorted(SANITIZER_FILES):
            text = _text(files[name], name)
            if "ERROR SUMMARY: 0 errors" not in text or not re.search(
                r"LEAK SUMMARY:\s+0 bytes leaked", text
            ):
                _fail(name, "does not contain the required zero-error/zero-leak result")
        _validate_host_runtime_log(
            files["compute-sanitizer-memcheck.log"],
            environment,
            gpu,
        )
        _validate_memory_log(
            files["compute-sanitizer-memory-memcheck.log"],
            "compute-sanitizer-memory-memcheck.log",
        )

    source = {
        "git_revision": source_revision,
        "git_dirty": False,
        "source_archive_sha256": source_archive_sha256,
        "release_binary_sha256": release_binary_sha256,
        "release_bundle_sha256": release_bundle_sha256,
        "release_image_sha256": release_image_sha256,
    }
    return source


def validate(
    evidence_dir: Path,
    *,
    source_revision: str,
    source_archive: Path,
    build_image_id: str,
    release_binary: Path,
    release_bundle: Path,
    release_image_id: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Validate all raw inputs and return bytes plus attestation source bindings."""

    files, environment = _read_evidence_directory(evidence_dir)
    source = _validate_bound_evidence(
        files,
        environment,
        source_revision=source_revision,
        source_archive=source_archive,
        build_image_id=build_image_id,
        release_binary=release_binary,
        release_bundle=release_bundle,
        release_image_id=release_image_id,
    )
    return files, source


def _attestation(source: dict[str, Any], raw_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": ATTESTATION_VERSION,
        "gate": GATE,
        "status": "passed",
        "source": source,
        "raw_evidence_sha256": raw_sha256,
        "checks": [
            {"id": check_id, "passed": True}
            for check_id in sorted(CHECK_IDS)
        ],
    }


def replay_raw_evidence(
    raw_evidence: Path,
    *,
    source_revision: str,
    source_archive: Path,
    build_image_id: str,
    release_binary: Path,
    release_bundle: Path,
    release_image_id: str,
) -> dict[str, Any]:
    """Replay preserved raw evidence against the exact release candidate."""

    files, environment, raw_sha256 = load_raw_evidence_archive(raw_evidence)
    source = _validate_bound_evidence(
        files,
        environment,
        source_revision=source_revision,
        source_archive=source_archive,
        build_image_id=build_image_id,
        release_binary=release_binary,
        release_bundle=release_bundle,
        release_image_id=release_image_id,
    )
    return _attestation(source, raw_sha256)


def _new_output_path(output: Path, label: str) -> None:
    try:
        metadata = output.parent.lstat()
    except OSError as error:
        _fail(label, f"cannot inspect output parent: {error}")
    if not stat.S_ISDIR(metadata.st_mode):
        _fail(label, "output parent must be a real directory")
    if output.exists() or output.is_symlink():
        _fail(label, "refusing to replace an existing path")


def _write_raw_archive(files: dict[str, bytes], output: Path) -> str:
    try:
        with output.open("xb") as destination:
            with tarfile.open(fileobj=destination, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(files):
                    info = tarfile.TarInfo(name)
                    info.size = len(files[name])
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(files[name]))
    except OSError as error:
        _fail("--raw-evidence", f"cannot create deterministic archive: {error}")
    return _sha256_file(output, "--raw-evidence", MAX_SOURCE_ARCHIVE_BYTES)


def produce(
    evidence_dir: Path,
    *,
    source_revision: str,
    source_archive: Path,
    build_image_id: str,
    release_binary: Path,
    release_bundle: Path,
    release_image_id: str,
    raw_evidence: Path,
    report: Path,
) -> dict[str, Any]:
    """Validate, package, and write a closed passing release-gate attestation."""

    if raw_evidence.resolve(strict=False) == report.resolve(strict=False):
        _fail("outputs", "--raw-evidence and --report must be different paths")
    try:
        evidence_root = evidence_dir.resolve(strict=True)
        output_paths = {
            "--raw-evidence": raw_evidence.resolve(strict=False),
            "--report": report.resolve(strict=False),
        }
    except (OSError, RuntimeError) as error:
        _fail("outputs", f"cannot resolve evidence/output paths: {error}")
    for label, output in output_paths.items():
        if output.is_relative_to(evidence_root):
            _fail(label, "must be outside --evidence-dir")
    _new_output_path(raw_evidence, "--raw-evidence")
    _new_output_path(report, "--report")
    files, source = validate(
        evidence_dir,
        source_revision=source_revision,
        source_archive=source_archive,
        build_image_id=build_image_id,
        release_binary=release_binary,
        release_bundle=release_bundle,
        release_image_id=release_image_id,
    )
    raw_sha256 = _write_raw_archive(files, raw_evidence)
    attestation = _attestation(source, raw_sha256)
    try:
        with report.open("xb") as destination:
            destination.write(canonical_json_bytes(attestation))
    except OSError as error:
        _fail("--report", f"cannot create attestation: {error}")
    return attestation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--build-image-id", required=True)
    parser.add_argument("--release-binary", required=True, type=Path)
    parser.add_argument("--release-bundle", required=True, type=Path)
    parser.add_argument("--release-image-id", required=True)
    parser.add_argument("--raw-evidence", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        produce(
            arguments.evidence_dir,
            source_revision=arguments.source_revision,
            source_archive=arguments.source_archive,
            build_image_id=arguments.build_image_id,
            release_binary=arguments.release_binary,
            release_bundle=arguments.release_bundle,
            release_image_id=arguments.release_image_id,
            raw_evidence=arguments.raw_evidence,
            report=arguments.report,
        )
    except CudaFaultEvidenceError as error:
        print(f"CUDA fault evidence rejected: {error}", file=sys.stderr)
        return 1
    print(f"CUDA fault evidence verified: {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
