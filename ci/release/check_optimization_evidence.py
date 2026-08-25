#!/usr/bin/env python3
"""Package and replay PR 15 optimizer correctness evidence on a CPU-only host.

The GPU runner supplies stdout/stderr plus the exact Rust test executables that
produced it.  This checker never executes those binaries.  It validates a
closed execution receipt, parses the asserted outcomes from the raw logs, and
stores everything in one canonical uncompressed USTAR archive.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import stat
import struct
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence

from release_common import ReleaseContractError, canonical_json_bytes, validate_binary


GATE_ID = "pr15-iteration-command-batch-exact-v1"
RECEIPT_VERSION = "rustinfer.optimizer-execution-receipt.v2"
EXPECTED_TOKENS = [
    4052,
    2025,
    284,
    965,
    6497,
    288,
    1492,
    418,
    260,
    16438,
    30,
    198,
    198,
    504,
    16438,
    314,
]

LOG_FILES = {
    "cuda-compile-only": "cuda-compile-only.log",
    "workspace-all-features-all-targets": "workspace-all-features-all-targets.log",
    "command-batch-lifecycle": "command-batch-lifecycle-gpu.log",
    "command-batch-resource-ledger": "command-batch-primitives-gpu.log",
    "smollm2-multi-step-greedy-exact": "iteration-command-batch-model-parity-gpu.log",
}
TEST_BINARIES = {
    "command-batch-lifecycle": "host-runtime-gpu-test",
    "command-batch-resource-ledger": "primitives-gpu-test",
    "smollm2-multi-step-greedy-exact": "llama-batch-gpu-test",
}
COMPILE_LOG_FILES = {
    "compile-command-batch-lifecycle": "command-batch-lifecycle-build.log",
    "compile-command-batch-resource-ledger": "command-batch-resource-ledger-build.log",
    "compile-smollm2-multi-step-greedy-exact": "smollm2-multi-step-greedy-exact-build.log",
}
REPORT_FILE = "optimization-correctness-report.json"
RECEIPT_FILE = "run-receipt.json"
CHECKSUM_FILE = "SHA256SUMS"
INPUT_FILES = (
    set(LOG_FILES.values())
    | set(COMPILE_LOG_FILES.values())
    | set(TEST_BINARIES.values())
    | {RECEIPT_FILE}
)
RAW_FILES = INPUT_FILES | {REPORT_FILE, CHECKSUM_FILE}


def _compile_argv(package: str, test_target: str, target_dir: str) -> list[str]:
    return [
        "cargo",
        "test",
        "--locked",
        "--offline",
        "--package",
        package,
        "--no-default-features",
        "--features",
        "cuda",
        "--test",
        test_target,
        "--no-run",
        "--message-format=json-render-diagnostics",
        "--color",
        "never",
        "--target-dir",
        target_dir,
    ]


TEST_SUBJECTS: dict[str, dict[str, str]] = {
    "host-runtime-gpu-test": {
        "cargo_test_target": "host_runtime_gpu",
        "compile_command_id": "compile-command-batch-lifecycle",
        "execute_command_id": "command-batch-lifecycle",
        "compile_log": COMPILE_LOG_FILES["compile-command-batch-lifecycle"],
        "package": "rustinfer-cuda",
        "target_dir": "/workspace/target/optimizer-evidence/command-batch-lifecycle",
        "test_name": "command_batch_proxy_is_one_shot_and_drop_restores_stream_use",
    },
    "primitives-gpu-test": {
        "cargo_test_target": "primitives_gpu",
        "compile_command_id": "compile-command-batch-resource-ledger",
        "execute_command_id": "command-batch-resource-ledger",
        "compile_log": COMPILE_LOG_FILES["compile-command-batch-resource-ledger"],
        "package": "rustinfer-cuda",
        "target_dir": "/workspace/target/optimizer-evidence/command-batch-resource-ledger",
        "test_name": "command_batch_releases_multi_primitive_resource_ledger_after_validation_error",
    },
    "llama-batch-gpu-test": {
        "cargo_test_target": "llama_batch_gpu",
        "compile_command_id": "compile-smollm2-multi-step-greedy-exact",
        "execute_command_id": "smollm2-multi-step-greedy-exact",
        "compile_log": COMPILE_LOG_FILES["compile-smollm2-multi-step-greedy-exact"],
        "package": "rustinfer-runtime",
        "target_dir": "/workspace/target/optimizer-evidence/smollm2-multi-step-greedy-exact",
        "test_name": "iteration_batch_completion_matches_per_operation_multi_step_greedy_exactly",
    },
}

EXPECTED_COMMANDS: dict[str, list[str]] = {
    "cuda-compile-only": ["/bin/sh", "ci/verify_python_free_cuda.sh"],
    "workspace-all-features-all-targets": [
        "cargo",
        "test",
        "--workspace",
        "--all-features",
        "--all-targets",
        "--locked",
        "--offline",
        "--color",
        "never",
    ],
    "compile-command-batch-lifecycle": _compile_argv(
        "rustinfer-cuda",
        "host_runtime_gpu",
        TEST_SUBJECTS["host-runtime-gpu-test"]["target_dir"],
    ),
    "command-batch-lifecycle": [
        "/evidence/host-runtime-gpu-test",
        "command_batch_proxy_is_one_shot_and_drop_restores_stream_use",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
        "--color",
        "never",
    ],
    "compile-command-batch-resource-ledger": _compile_argv(
        "rustinfer-cuda",
        "primitives_gpu",
        TEST_SUBJECTS["primitives-gpu-test"]["target_dir"],
    ),
    "command-batch-resource-ledger": [
        "/evidence/primitives-gpu-test",
        "command_batch_releases_multi_primitive_resource_ledger_after_validation_error",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
        "--color",
        "never",
    ],
    "compile-smollm2-multi-step-greedy-exact": _compile_argv(
        "rustinfer-runtime",
        "llama_batch_gpu",
        TEST_SUBJECTS["llama-batch-gpu-test"]["target_dir"],
    ),
    "smollm2-multi-step-greedy-exact": [
        "/evidence/llama-batch-gpu-test",
        "iteration_batch_completion_matches_per_operation_multi_step_greedy_exactly",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
        "--color",
        "never",
    ],
}
COMMAND_LOG_FILES = {
    "cuda-compile-only": LOG_FILES["cuda-compile-only"],
    "workspace-all-features-all-targets": LOG_FILES[
        "workspace-all-features-all-targets"
    ],
    "compile-command-batch-lifecycle": COMPILE_LOG_FILES[
        "compile-command-batch-lifecycle"
    ],
    "command-batch-lifecycle": LOG_FILES["command-batch-lifecycle"],
    "compile-command-batch-resource-ledger": COMPILE_LOG_FILES[
        "compile-command-batch-resource-ledger"
    ],
    "command-batch-resource-ledger": LOG_FILES["command-batch-resource-ledger"],
    "compile-smollm2-multi-step-greedy-exact": COMPILE_LOG_FILES[
        "compile-smollm2-multi-step-greedy-exact"
    ],
    "smollm2-multi-step-greedy-exact": LOG_FILES[
        "smollm2-multi-step-greedy-exact"
    ],
}
COMMAND_TEST_BINARIES = {
    "cuda-compile-only": None,
    "workspace-all-features-all-targets": None,
    "compile-command-batch-lifecycle": "host-runtime-gpu-test",
    "command-batch-lifecycle": "host-runtime-gpu-test",
    "compile-command-batch-resource-ledger": "primitives-gpu-test",
    "command-batch-resource-ledger": "primitives-gpu-test",
    "compile-smollm2-multi-step-greedy-exact": "llama-batch-gpu-test",
    "smollm2-multi-step-greedy-exact": "llama-batch-gpu-test",
}
BASE_ENVIRONMENT = {
    "CARGO_NET_OFFLINE": "true",
    "CARGO_TERM_COLOR": "never",
    "CUDA_HOME": "/usr/local/cuda",
    "CUDAToolkit_ROOT": "/usr/local/cuda",
    "RUSTINFER_CUDA_ARCHITECTURES": "89",
    "RUSTUP_TOOLCHAIN": "1.85.0-x86_64-unknown-linux-gnu",
}

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._+-]*)$")
ABSOLUTE_EVIDENCE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/+@=-]+$")
ANSI_RE = re.compile(rb"\x1b\[[0-9;]*[A-Za-z]")
TEST_SUMMARY_RE = re.compile(
    r"test result: ok\. (?P<passed>[0-9]+) passed; (?P<failed>[0-9]+) failed; "
    r"(?P<ignored>[0-9]+) ignored; (?P<measured>[0-9]+) measured; "
    r"(?P<filtered>[0-9]+) filtered out;"
)
WORKSPACE_HEADING_RE = re.compile(
    r"(?m)^\s*(?:Running (?:unittests|tests/)[^\r\n]+|Doc-tests [A-Za-z0-9_-]+)\r?$"
)
EXACT_SUMMARY_END_RE = re.compile(
    r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; "
    r"(?P<filtered>[0-9]+) filtered out; finished in [0-9]+(?:\.[0-9]+)?s"
    r"\r?\n\r?\n\Z"
)
PARITY_RE = re.compile(
    r"pr15-execution-completion-parity schema_version=1 decode_steps=(?P<steps>[0-9]+) "
    r"committed_iterations=(?P<iterations>[0-9]+) raw_logit_mismatches=(?P<logits>[0-9]+) "
    r"token_id_mismatches=(?P<tokens>[0-9]+) cuda_live_allocation_delta=(?P<hot>-?[0-9]+) "
    r"owner_close_live_allocation_count=(?P<close>[0-9]+) "
    r"generated_token_ids=\[(?P<ids>[0-9, ]+)\] status=passed"
)

MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + 4 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024


class OptimizationEvidenceError(ValueError):
    """The optimizer evidence is incomplete, unsafe, or not replayable."""


def _fail(path: str, message: str) -> NoReturn:
    raise OptimizationEvidenceError(f"{path}: {message}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON", f"duplicate key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail("JSON", f"non-finite number {value!r}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail("JSON", f"non-finite number {value!r}")
    return parsed


def _json(contents: bytes, label: str) -> dict[str, Any]:
    if not contents or len(contents) > MAX_JSON_BYTES:
        _fail(label, "must be nonempty and within the JSON size bound")
    try:
        text = contents.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(label, f"is not strict UTF-8 JSON: {error}")
    if not isinstance(value, dict):
        _fail(label, "must be a JSON object")
    return value


def _closed(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        _fail(label, f"closed field set mismatch: {observed}")
    return value


def _regular(path: Path, label: str, maximum: int = MAX_FILE_BYTES) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        _fail(label, f"cannot inspect {path}: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(label, "must be a regular file, not a link or device")
    if metadata.st_size <= 0 or metadata.st_size > maximum:
        _fail(label, f"must be nonempty and no larger than {maximum} bytes")
    return path


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _sha256_file(path: Path, label: str, maximum: int = MAX_FILE_BYTES) -> str:
    path = _regular(path, label, maximum)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _fail(label, f"cannot hash {path}: {error}")
    return digest.hexdigest()


def _text(contents: bytes, label: str) -> str:
    try:
        return ANSI_RE.sub(b"", contents).decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(label, f"is not UTF-8 text: {error}")


def _validate_test_binary(contents: bytes, filename: str, markers: Sequence[str]) -> None:
    try:
        validate_binary(contents)
    except ReleaseContractError as error:
        _fail(filename, f"is not a valid Linux x86_64 dynamic ELF: {error}")
    _validate_executable_elf(contents, filename)
    for marker in markers:
        if marker.encode("ascii") not in contents:
            _fail(filename, f"does not contain reviewed test marker {marker!r}")


def _validate_executable_elf(contents: bytes, label: str) -> None:
    """Reject dynamic ELF fixtures that cannot actually be executed."""

    if len(contents) < 64 or contents[:4] != b"\x7fELF":
        _fail(label, "is not an ELF file")
    if contents[4:7] != bytes((2, 1, 1)) or contents[7] not in (0, 3):
        _fail(label, "must be a 64-bit little-endian Linux/System-V ELF")
    try:
        header = struct.unpack_from("<HHIQQQIHHHHHH", contents, 16)
    except struct.error as error:
        _fail(label, f"has a truncated ELF header: {error}")
    elf_type, machine, version = header[:3]
    entry, phoff, phentsize, phnum = header[3], header[4], header[8], header[9]
    if elf_type not in (2, 3) or machine != 62 or version != 1 or entry == 0:
        _fail(label, "must be an executable Linux x86-64 ET_EXEC/ET_DYN ELF")
    if phentsize != 56 or phnum == 0 or phoff + phentsize * phnum > len(contents):
        _fail(label, "has an invalid ELF program-header table")
    executable_load = False
    for index in range(phnum):
        try:
            segment = struct.unpack_from("<IIQQQQQQ", contents, phoff + index * phentsize)
        except struct.error as error:
            _fail(label, f"has a truncated program header: {error}")
        segment_type, flags, offset, file_size = segment[0], segment[1], segment[2], segment[5]
        if offset + file_size > len(contents):
            _fail(label, "contains an out-of-range ELF segment")
        if segment_type == 1 and flags & 1 and file_size > 0:
            executable_load = True
    if not executable_load:
        _fail(label, "does not contain an executable PT_LOAD segment")


def _validate_summaries(
    text: str,
    label: str,
    *,
    minimum: int = 1,
    expected_process_failure_count: int = 0,
) -> list[re.Match[str]]:
    summaries = list(TEST_SUMMARY_RE.finditer(text))
    if len(summaries) < minimum:
        _fail(label, "does not contain a complete Cargo test summary")
    for summary in summaries:
        if int(summary.group("failed")) != 0:
            _fail(label, "contains a failed Cargo test summary")
    failure_markers = (
        "test result: FAILED",
        "fatal runtime error",
        "panicked at",
        "error: test failed",
        "SIGSEGV",
        "signal: 11",
    )
    for marker in failure_markers:
        if marker in text:
            _fail(label, f"contains failing test-run marker {marker!r}")
    process_marker = "process didn't exit successfully"
    if text.count(process_marker) != expected_process_failure_count:
        _fail(
            label,
            "contains an unexpected count of Cargo process-failure markers",
        )
    return summaries


def _single_marker(text: str, marker: str, label: str) -> None:
    if text.count(marker) != 1:
        _fail(label, f"must contain exactly one {marker!r} marker")


def _validate_exact_test_log(
    text: str,
    label: str,
    *,
    test_heading: str,
    semantic_marker: str,
    filtered: int,
) -> None:
    running = "running 1 test\n"
    ok_line = "\nok\n"
    _single_marker(text, running, label)
    _single_marker(text, test_heading, label)
    _single_marker(text, semantic_marker, label)
    _single_marker(text, ok_line, label)
    summary = EXACT_SUMMARY_END_RE.search(text)
    if summary is None or int(summary.group("filtered")) != filtered:
        _fail(label, "does not end with the exact passing libtest summary")
    positions = (
        text.index(running),
        text.index(test_heading),
        text.index(semantic_marker),
        text.index(ok_line),
        summary.start(),
    )
    if positions != tuple(sorted(positions)) or len(set(positions)) != len(positions):
        _fail(label, "test execution records are not in the reviewed order")


def _parse_logs(files: Mapping[str, bytes], report: Mapping[str, Any]) -> dict[str, str]:
    texts = {
        test_id: _text(files[filename], filename)
        for test_id, filename in LOG_FILES.items()
    }

    compile_log = texts["cuda-compile-only"]
    invalid_cuda_root_marker = (
        "error: rustinfer-cuda native build failed: "
        "CUDAToolkit_ROOT=/definitely/missing/rustinfer-cuda is not a directory"
    )
    for marker in (
        "rustc 1.85.0",
        "cargo 1.85.0",
        "Cuda compilation tools, release 12.8, V12.8.93",
        "test native_symbols_link_without_device_initialization ... ok",
        "rustinfer 0.1.0 (server=true, cuda=true, cuda_abi=1)",
        invalid_cuda_root_marker,
        "artifact=target/release/rustinfer\n",
        "artifact=target/release/rustinfer-profile\n",
        "Python-free CUDA production/profile compile, C ABI link, tensor memory, version, and dependency smoke passed",
    ):
        if marker not in compile_log:
            _fail(LOG_FILES["cuda-compile-only"], f"missing reviewed marker {marker!r}")
    _single_marker(
        compile_log,
        invalid_cuda_root_marker,
        LOG_FILES["cuda-compile-only"],
    )
    _validate_summaries(
        compile_log,
        LOG_FILES["cuda-compile-only"],
        expected_process_failure_count=1,
    )

    workspace = texts["workspace-all-features-all-targets"]
    summaries = _validate_summaries(
        workspace,
        LOG_FILES["workspace-all-features-all-targets"],
        minimum=20,
    )
    if not any(int(row.group("passed")) > 0 for row in summaries):
        _fail(LOG_FILES["workspace-all-features-all-targets"], "contains no executed tests")
    headings = WORKSPACE_HEADING_RE.findall(workspace)
    if len(headings) < 20 or len(set(headings)) != len(headings):
        _fail(
            LOG_FILES["workspace-all-features-all-targets"],
            "must record at least 20 distinct Cargo test-target headings",
        )
    for marker in (
        "command_batch_proxy_is_one_shot_and_drop_restores_stream_use ... ignored",
        "command_batch_releases_multi_primitive_resource_ledger_after_validation_error ... ignored",
        "iteration_batch_completion_matches_per_operation_multi_step_greedy_exactly ... ignored",
    ):
        if marker not in workspace:
            _fail(LOG_FILES["workspace-all-features-all-targets"], f"missing inventory marker {marker!r}")

    lifecycle = texts["command-batch-lifecycle"]
    lifecycle_test = "test command_batch_proxy_is_one_shot_and_drop_restores_stream_use ..."
    lifecycle_marker = (
        "pr16-command-batch-lifecycle schema_version=1 one_shot_finish=true "
        "drop_restores_stream=true status=passed"
    )
    _validate_summaries(lifecycle, LOG_FILES["command-batch-lifecycle"])
    _validate_exact_test_log(
        lifecycle,
        LOG_FILES["command-batch-lifecycle"],
        test_heading=lifecycle_test,
        semantic_marker=lifecycle_marker,
        filtered=7,
    )

    ledger = texts["command-batch-resource-ledger"]
    ledger_test = (
        "test command_batch_releases_multi_primitive_resource_ledger_after_validation_error ..."
    )
    ledger_marker = (
        "pr16-command-batch-resource-ledger schema_version=1 "
        "validation_fail_closed=true queued_chain_raw_byte_mismatches=0 "
        "cuda_live_allocation_delta=0 stream_reuse_after_finish=true "
        "owner_close_live_allocation_count=0 status=passed"
    )
    _validate_summaries(ledger, LOG_FILES["command-batch-resource-ledger"])
    _validate_exact_test_log(
        ledger,
        LOG_FILES["command-batch-resource-ledger"],
        test_heading=ledger_test,
        semantic_marker=ledger_marker,
        filtered=5,
    )

    parity = texts["smollm2-multi-step-greedy-exact"]
    parity_test = (
        "test iteration_batch_completion_matches_per_operation_multi_step_greedy_exactly ..."
    )
    matches = list(PARITY_RE.finditer(parity))
    if len(matches) != 1:
        _fail(LOG_FILES["smollm2-multi-step-greedy-exact"], "must contain one closed parity marker")
    _validate_summaries(parity, LOG_FILES["smollm2-multi-step-greedy-exact"])
    _validate_exact_test_log(
        parity,
        LOG_FILES["smollm2-multi-step-greedy-exact"],
        test_heading=parity_test,
        semantic_marker=matches[0].group(0),
        filtered=6,
    )
    match = matches[0]
    ids = [int(value) for value in match.group("ids").split(", ")]
    derived = {
        "decode_steps": int(match.group("steps")),
        "committed_iterations": int(match.group("iterations")),
        "raw_logit_mismatches": int(match.group("logits")),
        "token_id_mismatches": int(match.group("tokens")),
        "cuda_live_allocation_delta": int(match.group("hot")),
        "owner_close_live_allocation_count": int(match.group("close")),
        "generated_token_ids": ids,
    }
    expected_derived = {
        "decode_steps": 16,
        "committed_iterations": 16,
        "raw_logit_mismatches": 0,
        "token_id_mismatches": 0,
        "cuda_live_allocation_delta": 0,
        "owner_close_live_allocation_count": 0,
        "generated_token_ids": EXPECTED_TOKENS,
    }
    if derived != expected_derived:
        _fail(LOG_FILES["smollm2-multi-step-greedy-exact"], "parity marker differs from reviewed E0 result")

    tests = report.get("tests")
    if not isinstance(tests, list):
        _fail(REPORT_FILE, "tests must be an array")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(tests):
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            _fail(REPORT_FILE, f"tests[{index}] is invalid")
        test_id = value["id"]
        if test_id in by_id:
            _fail(REPORT_FILE, f"duplicate test id {test_id!r}")
        by_id[test_id] = value
    if set(by_id) != set(LOG_FILES):
        _fail(REPORT_FILE, f"exact test inventory mismatch: {sorted(by_id)}")
    for test_id, filename in LOG_FILES.items():
        if by_id[test_id].get("result") != "passed":
            _fail(REPORT_FILE, f"test {test_id!r} did not pass")
        if by_id[test_id].get("log_sha256") != _sha256(files[filename]):
            _fail(REPORT_FILE, f"test {test_id!r} log digest differs from raw bytes")
    for field, value in expected_derived.items():
        if by_id["smollm2-multi-step-greedy-exact"].get(field) != value:
            _fail(REPORT_FILE, f"parity field {field!r} differs from raw marker")
    for field in ("one_shot_finish", "drop_restores_stream"):
        if by_id["command-batch-lifecycle"].get(field) is not True:
            _fail(REPORT_FILE, f"lifecycle field {field!r} differs from raw marker")
    expected_ledger = {
        "validation_fail_closed": True,
        "queued_chain_raw_byte_mismatches": 0,
        "cuda_live_allocation_delta": 0,
        "stream_reuse_after_finish": True,
        "owner_close_live_allocation_count": 0,
    }
    for field, value in expected_ledger.items():
        if by_id["command-batch-resource-ledger"].get(field) != value:
            _fail(REPORT_FILE, f"resource-ledger field {field!r} differs from raw marker")
    return {test_id: _sha256(files[name]) for test_id, name in LOG_FILES.items()}


def _validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != 1 or report.get("gate_id") != GATE_ID:
        _fail(REPORT_FILE, "schema/gate mismatch")
    if report.get("status") != "passed" or report.get("semantic_class") != "E0":
        _fail(REPORT_FILE, "must be a passed E0 report")
    if not isinstance(report.get("recorded_at_utc"), str) or UTC_RE.fullmatch(
        report["recorded_at_utc"]
    ) is None:
        _fail(REPORT_FILE, "recorded_at_utc must be a UTC second timestamp")


def _validate_cargo_provenance(
    files: Mapping[str, bytes], subjects: Mapping[str, Mapping[str, Any]]
) -> None:
    """Bind each copied ELF to one fresh Cargo compiler-artifact record."""

    for filename, subject in subjects.items():
        spec = TEST_SUBJECTS[filename]
        label = spec["compile_log"]
        text = _text(files[label], label)
        for marker in (
            "error: could not compile",
            "error: test failed",
            "process didn't exit successfully",
            "fatal runtime error",
        ):
            if marker in text:
                _fail(label, f"contains failing Cargo marker {marker!r}")
        target_artifacts: list[dict[str, Any]] = []
        build_finished: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.startswith("{"):
                continue
            event = _json(line.encode("utf-8"), f"{label}:{line_number}")
            reason = event.get("reason")
            if reason == "build-finished":
                build_finished.append(event)
            if reason != "compiler-artifact":
                continue
            target = event.get("target")
            if isinstance(target, dict) and target.get("name") == spec[
                "cargo_test_target"
            ]:
                target_artifacts.append(event)
        if len(build_finished) != 1 or build_finished[0].get("success") is not True:
            _fail(label, "must contain exactly one successful Cargo build-finished record")
        if (
            len(target_artifacts) != 1
            or target_artifacts[0].get("executable")
            != subject["cargo_executable_path"]
        ):
            _fail(
                label,
                f"must contain exactly one matching compiler-artifact for {filename}",
            )
        artifact = target_artifacts[0]
        target = artifact["target"]
        profile = artifact.get("profile")
        if (
            target.get("kind") != ["test"]
            or target.get("crate_types") != ["bin"]
            or not isinstance(profile, dict)
            or profile.get("test") is not True
            or artifact.get("fresh") is not False
            or artifact.get("features") != ["cuda"]
        ):
            _fail(
                label,
                f"compiler-artifact for {filename} is not a fresh CUDA test executable",
            )


def _validate_receipt(
    receipt: Mapping[str, Any],
    report: Mapping[str, Any],
    files: Mapping[str, bytes],
    *,
    source_revision: str,
    source_archive_sha256: str,
    build_image_id: str,
    profile_binary_sha256: str,
) -> None:
    row = _closed(
        receipt,
        {
            "schema_version",
            "status",
            "source",
            "build",
            "gpu",
            "model",
            "profile_binary_sha256",
            "subjects",
            "commands",
        },
        RECEIPT_FILE,
    )
    if row["schema_version"] != RECEIPT_VERSION or row["status"] != "completed":
        _fail(RECEIPT_FILE, "schema/status mismatch")
    expected_source = {
        "git_commit": source_revision,
        "git_dirty": False,
        "archive_sha256": source_archive_sha256,
    }
    if row["source"] != expected_source or report.get("source") != expected_source:
        _fail(RECEIPT_FILE, "source does not bind the supplied archive")
    image_match = IMAGE_RE.fullmatch(build_image_id)
    if image_match is None:
        _fail("--build-image-id", "must be sha256:<lowercase digest>")
    report_build = report.get("build")
    if not isinstance(report_build, dict):
        _fail(REPORT_FILE, "build is invalid")
    if report_build.get("container_image_sha256") != image_match.group(1):
        _fail(REPORT_FILE, "build image differs from the supplied immutable image ID")
    if row["build"] != report_build:
        _fail(RECEIPT_FILE, "build differs from the report")
    for key, value in {
        "network": "none",
        "cargo_locked": True,
        "cargo_offline": True,
        "rustc": "1.85.0",
        "cuda_toolkit": "12.8.93",
        "cuda_architecture": "89",
    }.items():
        if report_build.get(key) != value:
            _fail(REPORT_FILE, f"build.{key} must be {value!r}")
    if row["gpu"] != report.get("gpu") or row["model"] != report.get("model"):
        _fail(RECEIPT_FILE, "GPU/model receipt differs from the submitted report")
    if row["profile_binary_sha256"] != profile_binary_sha256:
        _fail(RECEIPT_FILE, "profile binary differs from the supplied executable")

    subjects = _closed(
        row["subjects"], set(TEST_SUBJECTS), f"{RECEIPT_FILE}.subjects"
    )
    for filename, specification in TEST_SUBJECTS.items():
        subject = _closed(
            subjects[filename],
            {
                "sha256",
                "size",
                "cargo_test_target",
                "cargo_executable_path",
                "cargo_executable_sha256",
                "copied_executable_path",
                "compile_command_id",
                "execute_command_id",
            },
            f"{RECEIPT_FILE}.subjects.{filename}",
        )
        cargo_path = subject["cargo_executable_path"]
        expected_cargo_prefix = (
            f"{specification['target_dir']}/debug/deps/"
            f"{specification['cargo_test_target']}-"
        )
        file_sha256 = _sha256(files[filename])
        if (
            not isinstance(cargo_path, str)
            or ABSOLUTE_EVIDENCE_PATH_RE.fullmatch(cargo_path) is None
            or not cargo_path.startswith(expected_cargo_prefix)
            or subject["sha256"] != file_sha256
            or subject["cargo_executable_sha256"] != file_sha256
            or subject["copied_executable_path"] != f"/evidence/{filename}"
            or subject["cargo_test_target"] != specification["cargo_test_target"]
            or subject["compile_command_id"] != specification["compile_command_id"]
            or subject["execute_command_id"] != specification["execute_command_id"]
            or isinstance(subject["size"], bool)
            or not isinstance(subject["size"], int)
            or subject["size"] != len(files[filename])
        ):
            _fail(RECEIPT_FILE, f"subject differs from raw test executable: {filename}")

    commands = row["commands"]
    if not isinstance(commands, list) or len(commands) != len(EXPECTED_COMMANDS):
        _fail(RECEIPT_FILE, "exact command inventory is required")
    observed: dict[str, Mapping[str, Any]] = {}
    command_order: list[str] = []
    for index, value in enumerate(commands):
        command = _closed(
            value,
            {"id", "argv", "environment", "exit_code", "log", "test_binary"},
            f"{RECEIPT_FILE}.commands[{index}]",
        )
        command_id = command["id"]
        if not isinstance(command_id, str) or command_id in observed:
            _fail(RECEIPT_FILE, "command id is invalid or duplicated")
        observed[command_id] = command
        command_order.append(command_id)
    if set(observed) != set(EXPECTED_COMMANDS):
        _fail(RECEIPT_FILE, f"command id set mismatch: {sorted(observed)}")
    if command_order != list(EXPECTED_COMMANDS):
        _fail(RECEIPT_FILE, "commands are not in the reviewed execution order")
    for command_id, argv in EXPECTED_COMMANDS.items():
        command = observed[command_id]
        expected_environment = dict(BASE_ENVIRONMENT)
        if command_id == "smollm2-multi-step-greedy-exact":
            expected_environment["RUSTINFER_REAL_CHECKPOINT"] = "/model"
        expected_binary = COMMAND_TEST_BINARIES[command_id]
        if command != {
            "id": command_id,
            "argv": argv,
            "environment": expected_environment,
            "exit_code": 0,
            "log": COMMAND_LOG_FILES[command_id],
            "test_binary": expected_binary,
        }:
            _fail(RECEIPT_FILE, f"command {command_id!r} differs from the reviewed invocation")
    _validate_cargo_provenance(files, subjects)


def _parse_checksums(contents: bytes) -> dict[str, str]:
    try:
        text = contents.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(CHECKSUM_FILE, f"is not ASCII: {error}")
    if not text.endswith("\n"):
        _fail(CHECKSUM_FILE, "must end in a newline")
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = CHECKSUM_RE.fullmatch(line)
        if match is None:
            _fail(CHECKSUM_FILE, f"invalid checksum line {line!r}")
        digest, name = match.groups()
        if name in result:
            _fail(CHECKSUM_FILE, f"duplicate path {name!r}")
        result[name] = digest
    expected = RAW_FILES - {CHECKSUM_FILE}
    if set(result) != expected or list(result) != sorted(result):
        _fail(CHECKSUM_FILE, f"closed sorted inventory mismatch: {sorted(result)}")
    return result


def _validate_raw_files(files: Mapping[str, bytes]) -> None:
    if set(files) != RAW_FILES:
        _fail("raw evidence", f"closed inventory mismatch: {sorted(files)}")
    if sum(len(value) for value in files.values()) > MAX_TOTAL_BYTES:
        _fail("raw evidence", "exceeds the total size bound")
    checksums = _parse_checksums(files[CHECKSUM_FILE])
    for name, digest in checksums.items():
        if _sha256(files[name]) != digest:
            _fail(CHECKSUM_FILE, f"digest mismatch for {name}")


def load_raw_evidence_archive(path: Path) -> tuple[dict[str, bytes], str]:
    """Read a canonical closed USTAR archive without extracting it."""

    raw_path = _regular(path, "--raw-evidence", MAX_ARCHIVE_BYTES)
    try:
        archive_bytes = raw_path.read_bytes()
    except OSError as error:
        _fail("--raw-evidence", f"cannot read archive: {error}")
    files: dict[str, bytes] = {}
    expected_offset = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            if archive.pax_headers:
                _fail("--raw-evidence", "global PAX headers are forbidden")
            members = archive.getmembers()
            if [member.name for member in members] != sorted(RAW_FILES):
                _fail("--raw-evidence", "exact bytewise-sorted member inventory is required")
            for member in members:
                name = member.name
                if (
                    name in files
                    or name not in RAW_FILES
                    or "/" in name
                    or PurePosixPath(name).name != name
                    or not member.isreg()
                ):
                    _fail("--raw-evidence", f"unsafe or duplicate member {name!r}")
                expected_mode = 0o755 if name in TEST_BINARIES.values() else 0o644
                if (
                    member.pax_headers
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != "root"
                    or member.gname != "root"
                    or member.mode != expected_mode
                    or member.mtime != 0
                    or member.linkname
                    or member.devmajor != 0
                    or member.devminor != 0
                ):
                    _fail("--raw-evidence", f"non-canonical metadata for {name}")
                if member.size <= 0 or member.size > MAX_FILE_BYTES:
                    _fail("--raw-evidence", f"invalid member size for {name}")
                if member.offset != expected_offset or member.offset_data != expected_offset + 512:
                    _fail("--raw-evidence", f"non-canonical layout for {name}")
                expected_info = tarfile.TarInfo(name)
                expected_info.size = member.size
                expected_info.mode = expected_mode
                expected_info.uid = 0
                expected_info.gid = 0
                expected_info.uname = "root"
                expected_info.gname = "root"
                expected_info.mtime = 0
                if archive_bytes[member.offset : member.offset_data] != expected_info.tobuf(
                    format=tarfile.USTAR_FORMAT
                ):
                    _fail("--raw-evidence", f"non-canonical USTAR header for {name}")
                data_end = member.offset_data + member.size
                padded_end = member.offset_data + ((member.size + 511) // 512) * 512
                if data_end > len(archive_bytes) or any(archive_bytes[data_end:padded_end]):
                    _fail("--raw-evidence", f"truncated or non-zero padded member {name}")
                files[name] = archive_bytes[member.offset_data:data_end]
                expected_offset = padded_end
    except OptimizationEvidenceError:
        raise
    except (OSError, tarfile.TarError, ValueError) as error:
        _fail("--raw-evidence", f"is not a canonical readable USTAR: {error}")
    canonical_size = ((expected_offset + 1024 + 10239) // 10240) * 10240
    if len(archive_bytes) != canonical_size or any(archive_bytes[expected_offset:]):
        _fail("--raw-evidence", "has non-canonical end-of-archive padding")
    _validate_raw_files(files)
    return files, _sha256(archive_bytes)


def replay_raw_evidence(
    raw_evidence: Path,
    *,
    report: Path,
    source_revision: str,
    source_archive_sha256: str,
    build_image_id: str,
    profile_binary: Path,
) -> dict[str, Any]:
    """Replay raw optimizer evidence against immutable candidate subjects."""

    if GIT_RE.fullmatch(source_revision) is None:
        _fail("--source-revision", "must be a lowercase 40-character Git SHA")
    if SHA_RE.fullmatch(source_archive_sha256) is None:
        _fail("--source-archive-sha256", "must be a lowercase SHA-256")
    report_path = _regular(report, "--report", MAX_JSON_BYTES)
    try:
        report_bytes = report_path.read_bytes()
    except OSError as error:
        _fail("--report", f"cannot read report: {error}")
    report_document = _json(report_bytes, "--report")
    _validate_report(report_document)
    profile_bytes = _regular(profile_binary, "--profile-binary").read_bytes()
    try:
        validate_binary(profile_bytes)
    except ReleaseContractError as error:
        _fail("--profile-binary", f"is not a valid Linux x86_64 dynamic ELF: {error}")
    _validate_executable_elf(profile_bytes, "--profile-binary")
    for marker in (GATE_ID, "per-operation", "iteration-batch"):
        if marker.encode("ascii") not in profile_bytes:
            _fail("--profile-binary", f"missing reviewed profile marker {marker!r}")
    profile_sha256 = _sha256(profile_bytes)

    files, raw_sha256 = load_raw_evidence_archive(raw_evidence)
    embedded_report = _json(files[REPORT_FILE], REPORT_FILE)
    if embedded_report != report_document or files[REPORT_FILE] != canonical_json_bytes(report_document):
        _fail(REPORT_FILE, "must be the canonical exact submitted report")

    binary_markers = {
        "host-runtime-gpu-test": [
            "command_batch_proxy_is_one_shot_and_drop_restores_stream_use",
            "pr16-command-batch-lifecycle",
        ],
        "primitives-gpu-test": [
            "command_batch_releases_multi_primitive_resource_ledger_after_validation_error",
            "pr16-command-batch-resource-ledger",
        ],
        "llama-batch-gpu-test": [
            "iteration_batch_completion_matches_per_operation_multi_step_greedy_exactly",
            "pr15-execution-completion-parity",
        ],
    }
    for filename, markers in binary_markers.items():
        _validate_test_binary(files[filename], filename, markers)
    receipt = _json(files[RECEIPT_FILE], RECEIPT_FILE)
    if files[RECEIPT_FILE] != canonical_json_bytes(receipt):
        _fail(RECEIPT_FILE, "must use canonical JSON encoding")
    _validate_receipt(
        receipt,
        report_document,
        files,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
        build_image_id=build_image_id,
        profile_binary_sha256=profile_sha256,
    )
    log_sha256 = _parse_logs(files, report_document)
    return {
        "report": report_document,
        "report_sha256": _sha256(report_bytes),
        "raw_evidence_sha256": raw_sha256,
        "profile_binary_sha256": profile_sha256,
        "build_image_sha256": build_image_id.removeprefix("sha256:"),
        "log_sha256": log_sha256,
        "test_binary_sha256": {
            name: _sha256(files[name]) for name in sorted(TEST_BINARIES.values())
        },
    }


def _read_input_directory(root: Path) -> dict[str, bytes]:
    try:
        metadata = root.lstat()
    except OSError as error:
        _fail("--evidence-dir", f"cannot inspect directory: {error}")
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("--evidence-dir", "must be a real directory, not a link")
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != INPUT_FILES or len(entries) != len(INPUT_FILES):
        _fail("--evidence-dir", f"closed input inventory mismatch: {sorted(entry.name for entry in entries)}")
    files: dict[str, bytes] = {}
    total = 0
    for name in sorted(INPUT_FILES):
        path = _regular(root / name, name)
        try:
            contents = path.read_bytes()
        except OSError as error:
            _fail(name, f"cannot read input: {error}")
        total += len(contents)
        if total > MAX_TOTAL_BYTES:
            _fail("--evidence-dir", "exceeds the total size bound")
        files[name] = contents
    return files


def _write_raw(files: Mapping[str, bytes], output: Path) -> None:
    parent = output.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        _fail("--raw-evidence", "output parent must be a real directory")
    if output.exists() or output.is_symlink():
        _fail("--raw-evidence", "refusing to replace an existing path")
    try:
        with output.open("xb") as destination:
            with tarfile.open(
                fileobj=destination, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive:
                for name in sorted(files):
                    info = tarfile.TarInfo(name)
                    info.size = len(files[name])
                    info.mode = 0o755 if name in TEST_BINARIES.values() else 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(files[name]))
    except OSError as error:
        _fail("--raw-evidence", f"cannot create archive: {error}")


def produce(
    evidence_dir: Path,
    *,
    report: Path,
    source_revision: str,
    source_archive_sha256: str,
    build_image_id: str,
    profile_binary: Path,
    raw_evidence: Path,
) -> dict[str, Any]:
    """Create a deterministic archive, then replay it before returning."""

    files = _read_input_directory(evidence_dir)
    report_bytes = _regular(report, "--report", MAX_JSON_BYTES).read_bytes()
    report_document = _json(report_bytes, "--report")
    if report_bytes != canonical_json_bytes(report_document):
        _fail("--report", "must use canonical JSON encoding")
    files[REPORT_FILE] = report_bytes
    checksums = "".join(
        f"{_sha256(files[name])}  {name}\n" for name in sorted(files)
    ).encode("ascii")
    files[CHECKSUM_FILE] = checksums
    _validate_raw_files(files)
    _write_raw(files, raw_evidence)
    return replay_raw_evidence(
        raw_evidence,
        report=report,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
        build_image_id=build_image_id,
        profile_binary=profile_binary,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--raw-evidence", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--build-image-id", required=True)
    parser.add_argument("--profile-binary", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.evidence_dir is None:
            result = replay_raw_evidence(
                args.raw_evidence,
                report=args.report,
                source_revision=args.source_revision,
                source_archive_sha256=args.source_archive_sha256,
                build_image_id=args.build_image_id,
                profile_binary=args.profile_binary,
            )
        else:
            result = produce(
                args.evidence_dir,
                report=args.report,
                source_revision=args.source_revision,
                source_archive_sha256=args.source_archive_sha256,
                build_image_id=args.build_image_id,
                profile_binary=args.profile_binary,
                raw_evidence=args.raw_evidence,
            )
    except (OSError, OptimizationEvidenceError) as error:
        print(f"optimization evidence rejected: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
