#!/usr/bin/env python3
"""Write canonical PR 15/16 optimizer execution evidence.

This producer consumes records emitted by the remote GPU runner.  It does not
run Cargo, CUDA, or a model.  In particular, it keeps the three Cargo
``--no-run`` build commands separate from the three executions of the copied
test ELFs, and binds every copied subject to the compiler-artifact path that
Cargo reported.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence


GATE_ID = "pr15-iteration-command-batch-exact-v1"
RECEIPT_VERSION = "rustinfer.optimizer-execution-receipt.v2"
COMMAND_RECORD_VERSION = "rustinfer.optimizer-command-log.v2"
SUBJECT_RECORD_VERSION = "rustinfer.optimizer-subjects.v2"

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_WEIGHTS_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
MODEL_TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
GPU_NAME = "NVIDIA GeForce RTX 4090"
GPU_UUID = "GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0"

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
COMPILE_LOG_FILES = {
    "compile-command-batch-lifecycle": "command-batch-lifecycle-build.log",
    "compile-command-batch-resource-ledger": "command-batch-resource-ledger-build.log",
    "compile-smollm2-multi-step-greedy-exact": "smollm2-multi-step-greedy-exact-build.log",
}

BASE_ENVIRONMENT = {
    "CARGO_NET_OFFLINE": "true",
    "CARGO_TERM_COLOR": "never",
    "CUDA_HOME": "/usr/local/cuda",
    "CUDAToolkit_ROOT": "/usr/local/cuda",
    "RUSTINFER_CUDA_ARCHITECTURES": "89",
    "RUSTUP_TOOLCHAIN": "1.85.0-x86_64-unknown-linux-gnu",
}


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

EXPECTED_COMMANDS: dict[str, dict[str, Any]] = {
    "cuda-compile-only": {
        "argv": ["/bin/sh", "ci/verify_python_free_cuda.sh"],
        "log": LOG_FILES["cuda-compile-only"],
        "test_binary": None,
    },
    "workspace-all-features-all-targets": {
        "argv": [
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
        "log": LOG_FILES["workspace-all-features-all-targets"],
        "test_binary": None,
    },
}
for _filename, _subject in TEST_SUBJECTS.items():
    _compile_id = _subject["compile_command_id"]
    _execute_id = _subject["execute_command_id"]
    EXPECTED_COMMANDS[_compile_id] = {
        "argv": _compile_argv(
            _subject["package"], _subject["cargo_test_target"], _subject["target_dir"]
        ),
        "log": _subject["compile_log"],
        "test_binary": _filename,
    }
    EXPECTED_COMMANDS[_execute_id] = {
        "argv": [
            f"/evidence/{_filename}",
            _subject["test_name"],
            "--ignored",
            "--exact",
            "--nocapture",
            "--test-threads=1",
            "--color",
            "never",
        ],
        "log": LOG_FILES[_execute_id],
        "test_binary": _filename,
    }

COMMAND_ORDER = [
    "cuda-compile-only",
    "workspace-all-features-all-targets",
    "compile-command-batch-lifecycle",
    "command-batch-lifecycle",
    "compile-command-batch-resource-ledger",
    "command-batch-resource-ledger",
    "compile-smollm2-multi-step-greedy-exact",
    "smollm2-multi-step-greedy-exact",
]

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
SAFE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/+@=-]+$")
PARITY_RE = re.compile(
    r"pr15-execution-completion-parity schema_version=1 decode_steps=(?P<steps>[0-9]+) "
    r"committed_iterations=(?P<iterations>[0-9]+) raw_logit_mismatches=(?P<logits>[0-9]+) "
    r"token_id_mismatches=(?P<tokens>[0-9]+) cuda_live_allocation_delta=(?P<hot>-?[0-9]+) "
    r"owner_close_live_allocation_count=(?P<close>[0-9]+) "
    r"generated_token_ids=\[(?P<ids>[0-9, ]+)\] status=passed"
)


class EvidenceWriterError(ValueError):
    """Remote records are incomplete, inconsistent, or unsafe."""


def _fail(path: str, message: str) -> NoReturn:
    raise EvidenceWriterError(f"{path}: {message}")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _regular(path: Path, label: str, *, executable: bool = False) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        _fail(label, f"cannot inspect {path}: {error}")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        _fail(label, "must be a nonempty regular file")
    if executable and metadata.st_mode & 0o111 == 0:
        _fail(label, "must have an executable mode bit")
    return path


def _sha256_file(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    with _regular(path, label).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(path: Path, label: str) -> str:
    try:
        return _regular(path, label).read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        _fail(label, f"must be UTF-8: {error}")


def _decode_record(value: str, label: str) -> str:
    try:
        raw = base64.b64decode(value, validate=True)
        text = raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        _fail(label, f"invalid base64 UTF-8 field: {error}")
    if not text or "\0" in text or "\n" in text or "\r" in text:
        _fail(label, "decoded fields must be nonempty single-line text")
    if base64.b64encode(raw).decode("ascii") != value:
        _fail(label, "base64 field is not canonical")
    return text


def parse_command_records(path: Path) -> list[dict[str, Any]]:
    lines = _text(path, "--command-records").splitlines()
    if not lines or lines[0] != COMMAND_RECORD_VERSION:
        _fail("--command-records", "version header mismatch")
    commands: list[dict[str, Any]] = []
    index = 1
    while index < len(lines):
        if not lines[index].startswith("BEGIN "):
            _fail("--command-records", f"expected BEGIN at line {index + 1}")
        command_id = _decode_record(lines[index][6:], "command id")
        index += 1
        fields: dict[str, str] = {}
        environment: dict[str, str] = {}
        argv: list[str] = []
        while index < len(lines) and lines[index] != "END":
            parts = lines[index].split(" ")
            kind = parts[0]
            if kind == "ENV" and len(parts) == 3:
                key = _decode_record(parts[1], "environment key")
                value = _decode_record(parts[2], f"environment {key}")
                if key in environment:
                    _fail("--command-records", f"duplicate environment key {key!r}")
                environment[key] = value
            elif kind == "ARG" and len(parts) == 2:
                argv.append(_decode_record(parts[1], "argv"))
            elif kind in {"LOG", "SUBJECT", "EXIT"} and len(parts) == 2:
                if kind in fields:
                    _fail("--command-records", f"duplicate {kind} field")
                fields[kind] = _decode_record(parts[1], kind.lower())
            else:
                _fail("--command-records", f"invalid record at line {index + 1}")
            index += 1
        if index >= len(lines) or lines[index] != "END":
            _fail("--command-records", f"unterminated command {command_id!r}")
        index += 1
        if set(fields) != {"LOG", "SUBJECT", "EXIT"} or not argv:
            _fail("--command-records", f"command {command_id!r} is incomplete")
        try:
            exit_code = int(fields["EXIT"])
        except ValueError:
            _fail("--command-records", f"command {command_id!r} has invalid exit code")
        commands.append(
            {
                "id": command_id,
                "argv": argv,
                "environment": environment,
                "exit_code": exit_code,
                "log": fields["LOG"],
                "test_binary": None if fields["SUBJECT"] == "-" else fields["SUBJECT"],
            }
        )
    if [row["id"] for row in commands] != COMMAND_ORDER:
        _fail("--command-records", "command inventory/order mismatch")
    for command in commands:
        expected = EXPECTED_COMMANDS[command["id"]]
        expected_environment = dict(BASE_ENVIRONMENT)
        if command["id"] == "smollm2-multi-step-greedy-exact":
            expected_environment["RUSTINFER_REAL_CHECKPOINT"] = "/model"
        if command != {
            "id": command["id"],
            "argv": expected["argv"],
            "environment": expected_environment,
            "exit_code": 0,
            "log": expected["log"],
            "test_binary": expected["test_binary"],
        }:
            _fail("--command-records", f"command {command['id']!r} differs from the reviewed invocation")
    return commands


def parse_subject_records(path: Path, evidence_dir: Path) -> dict[str, dict[str, Any]]:
    lines = _text(path, "--subject-records").splitlines()
    if not lines or lines[0] != SUBJECT_RECORD_VERSION:
        _fail("--subject-records", "version header mismatch")
    observed: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != 9:
            _fail("--subject-records", f"line {line_number} must have nine tab-separated fields")
        (
            filename,
            cargo_target,
            cargo_path,
            cargo_sha,
            copied_path,
            copied_sha,
            size_text,
            compile_id,
            execute_id,
        ) = fields
        if filename in observed or filename not in TEST_SUBJECTS:
            _fail("--subject-records", f"unsafe or duplicate subject {filename!r}")
        if SAFE_PATH_RE.fullmatch(cargo_path) is None or SAFE_PATH_RE.fullmatch(copied_path) is None:
            _fail("--subject-records", f"subject {filename!r} has an unsafe absolute path")
        if not SHA_RE.fullmatch(cargo_sha) or not SHA_RE.fullmatch(copied_sha):
            _fail("--subject-records", f"subject {filename!r} has an invalid digest")
        try:
            size = int(size_text)
        except ValueError:
            _fail("--subject-records", f"subject {filename!r} has an invalid size")
        expected = TEST_SUBJECTS[filename]
        if (
            cargo_target != expected["cargo_test_target"]
            or not cargo_path.startswith(expected["target_dir"] + "/")
            or copied_path != f"/evidence/{filename}"
            or compile_id != expected["compile_command_id"]
            or execute_id != expected["execute_command_id"]
            or cargo_sha != copied_sha
            or size <= 0
        ):
            _fail("--subject-records", f"subject {filename!r} provenance mismatch")
        copied = _regular(evidence_dir / filename, filename, executable=True)
        actual_sha = _sha256_file(copied, filename)
        if actual_sha != copied_sha or copied.stat().st_size != size:
            _fail("--subject-records", f"subject {filename!r} differs from copied bytes")
        observed[filename] = {
            "sha256": copied_sha,
            "size": size,
            "cargo_test_target": cargo_target,
            "cargo_executable_path": cargo_path,
            "cargo_executable_sha256": cargo_sha,
            "copied_executable_path": copied_path,
            "compile_command_id": compile_id,
            "execute_command_id": execute_id,
        }
    if list(observed) != list(TEST_SUBJECTS) or set(observed) != set(TEST_SUBJECTS):
        _fail("--subject-records", "closed subject inventory/order mismatch")
    return observed


def validate_cargo_provenance(
    evidence_dir: Path, subjects: Mapping[str, Mapping[str, Any]]
) -> None:
    for filename, subject in subjects.items():
        spec = TEST_SUBJECTS[filename]
        log_path = evidence_dir / spec["compile_log"]
        matches: list[dict[str, Any]] = []
        for line in _text(log_path, spec["compile_log"]).splitlines():
            if not line.startswith("{"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict) or value.get("reason") != "compiler-artifact":
                continue
            target = value.get("target")
            if not isinstance(target, dict) or target.get("name") != spec["cargo_test_target"]:
                continue
            if value.get("executable") == subject["cargo_executable_path"]:
                matches.append(value)
        if len(matches) != 1:
            _fail(spec["compile_log"], f"must contain exactly one matching compiler-artifact for {filename}")
        artifact = matches[0]
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
            _fail(spec["compile_log"], f"compiler-artifact for {filename} is not a fresh CUDA test executable")


def parse_gpu(path: Path) -> dict[str, Any]:
    try:
        rows = list(csv.reader(_text(path, "--gpu-csv").splitlines()))
    except csv.Error as error:
        _fail("--gpu-csv", f"invalid CSV: {error}")
    if len(rows) != 1 or len(rows[0]) != 6:
        _fail("--gpu-csv", "must contain exactly one six-column GPU row")
    name, uuid, pci_bus_id, memory, driver, compute = (value.strip() for value in rows[0])
    if name != GPU_NAME or uuid != GPU_UUID or compute != "8.9":
        _fail("--gpu-csv", "does not identify the designated server-4096 RTX 4090")
    if not re.fullmatch(r"[0-9]+", memory) or int(memory) < 24000:
        _fail("--gpu-csv", "VRAM inventory is invalid")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", driver):
        _fail("--gpu-csv", "driver version is invalid")
    if not re.fullmatch(r"[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]", pci_bus_id):
        _fail("--gpu-csv", "PCI bus ID is invalid")
    return {
        "model": name,
        "uuid": uuid,
        "pci_bus_id": pci_bus_id,
        "compute_capability": compute,
        "vram_mib": int(memory),
        "driver_version": driver,
    }


def _validate_execution_logs(evidence_dir: Path) -> dict[str, dict[str, Any]]:
    for filename in [*LOG_FILES.values(), *COMPILE_LOG_FILES.values()]:
        _regular(evidence_dir / filename, filename)
    lifecycle = _text(evidence_dir / LOG_FILES["command-batch-lifecycle"], "lifecycle log")
    lifecycle_marker = (
        "pr16-command-batch-lifecycle schema_version=1 one_shot_finish=true "
        "drop_restores_stream=true status=passed"
    )
    ledger = _text(evidence_dir / LOG_FILES["command-batch-resource-ledger"], "ledger log")
    ledger_marker = (
        "pr16-command-batch-resource-ledger schema_version=1 validation_fail_closed=true "
        "queued_chain_raw_byte_mismatches=0 cuda_live_allocation_delta=0 "
        "stream_reuse_after_finish=true owner_close_live_allocation_count=0 status=passed"
    )
    for label, text, marker in (
        ("lifecycle log", lifecycle, lifecycle_marker),
        ("ledger log", ledger, ledger_marker),
    ):
        if text.count(marker) != 1 or "test result: ok. 1 passed; 0 failed;" not in text:
            _fail(label, "does not contain one exact passing semantic marker")
    parity = _text(
        evidence_dir / LOG_FILES["smollm2-multi-step-greedy-exact"], "parity log"
    )
    matches = list(PARITY_RE.finditer(parity))
    if len(matches) != 1 or "test result: ok. 1 passed; 0 failed;" not in parity:
        _fail("parity log", "does not contain one exact passing parity marker")
    match = matches[0]
    derived = {
        "decode_steps": int(match.group("steps")),
        "committed_iterations": int(match.group("iterations")),
        "raw_logit_mismatches": int(match.group("logits")),
        "token_id_mismatches": int(match.group("tokens")),
        "cuda_live_allocation_delta": int(match.group("hot")),
        "owner_close_live_allocation_count": int(match.group("close")),
        "generated_token_ids": [int(value) for value in match.group("ids").split(", ")],
    }
    expected = {
        "decode_steps": 16,
        "committed_iterations": 16,
        "raw_logit_mismatches": 0,
        "token_id_mismatches": 0,
        "cuda_live_allocation_delta": 0,
        "owner_close_live_allocation_count": 0,
        "generated_token_ids": EXPECTED_TOKENS,
    }
    if derived != expected:
        _fail("parity log", "semantic values differ from the reviewed E0 result")
    return {
        "command-batch-lifecycle": {"one_shot_finish": True, "drop_restores_stream": True},
        "command-batch-resource-ledger": {
            "validation_fail_closed": True,
            "queued_chain_raw_byte_mismatches": 0,
            "cuda_live_allocation_delta": 0,
            "stream_reuse_after_finish": True,
            "owner_close_live_allocation_count": 0,
        },
        "smollm2-multi-step-greedy-exact": derived,
    }


def write_evidence(
    evidence_dir: Path,
    *,
    command_records: Path,
    subject_records: Path,
    gpu_csv: Path,
    report_path: Path,
    receipt_path: Path,
    source_revision: str,
    source_archive_sha256: str,
    build_image_id: str,
    profile_binary: Path,
    model_tree_sha256: str,
    recorded_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if GIT_RE.fullmatch(source_revision) is None:
        _fail("--source-revision", "must be a lowercase 40-character Git SHA")
    if SHA_RE.fullmatch(source_archive_sha256) is None:
        _fail("--source-archive-sha256", "must be a lowercase SHA-256")
    image_match = IMAGE_RE.fullmatch(build_image_id)
    if image_match is None:
        _fail("--build-image-id", "must be sha256:<lowercase digest>")
    if SHA_RE.fullmatch(model_tree_sha256) is None:
        _fail("--model-tree-sha256", "must be a lowercase SHA-256")
    if UTC_RE.fullmatch(recorded_at_utc) is None:
        _fail("--recorded-at-utc", "must be a UTC second timestamp")
    if not evidence_dir.is_dir() or evidence_dir.is_symlink():
        _fail("--evidence-dir", "must be a real directory")
    expected_inputs = {
        *LOG_FILES.values(),
        *COMPILE_LOG_FILES.values(),
        *TEST_SUBJECTS,
    }
    try:
        observed_inputs = {entry.name for entry in evidence_dir.iterdir()}
    except OSError as error:
        _fail("--evidence-dir", f"cannot list evidence inputs: {error}")
    if observed_inputs != expected_inputs:
        _fail(
            "--evidence-dir",
            f"closed pre-receipt inventory mismatch: {sorted(observed_inputs)}",
        )
    for output, label in ((report_path, "--report"), (receipt_path, "--receipt")):
        if output.exists() or output.is_symlink():
            _fail(label, "refusing to replace an existing path")
        if not output.parent.is_dir() or output.parent.is_symlink():
            _fail(label, "output parent must be a real directory")

    commands = parse_command_records(command_records)
    subjects = parse_subject_records(subject_records, evidence_dir)
    validate_cargo_provenance(evidence_dir, subjects)
    gpu = parse_gpu(gpu_csv)
    semantic = _validate_execution_logs(evidence_dir)
    profile_sha256 = _sha256_file(profile_binary, "--profile-binary")

    source = {
        "git_commit": source_revision,
        "git_dirty": False,
        "archive_sha256": source_archive_sha256,
    }
    build = {
        "container_image_sha256": image_match.group(1),
        "network": "none",
        "cargo_locked": True,
        "cargo_offline": True,
        "rustc": "1.85.0",
        "cuda_toolkit": "12.8.93",
        "cuda_architecture": "89",
    }
    model = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": "bf16",
        "manifest_sha256": model_tree_sha256,
        "weights_sha256": MODEL_WEIGHTS_SHA256,
        "tokenizer_sha256": MODEL_TOKENIZER_SHA256,
    }
    tests: list[dict[str, Any]] = []
    for test_id, filename in LOG_FILES.items():
        test = {
            "id": test_id,
            "result": "passed",
            "log_sha256": _sha256_file(evidence_dir / filename, filename),
        }
        test.update(semantic.get(test_id, {}))
        tests.append(test)
    report = {
        "schema_version": 1,
        "gate_id": GATE_ID,
        "recorded_at_utc": recorded_at_utc,
        "status": "passed",
        "semantic_class": "E0",
        "source": source,
        "build": build,
        "gpu": gpu,
        "model": model,
        "implementations": {
            "baseline": "per-operation",
            "candidate": "iteration-batch",
            "residual_rmsnorm": "separate",
            "rollback": "--execution-completion per-operation",
        },
        "tests": tests,
    }
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "status": "completed",
        "source": source,
        "build": build,
        "gpu": gpu,
        "model": model,
        "profile_binary_sha256": profile_sha256,
        "subjects": subjects,
        "commands": commands,
    }
    try:
        report_path.write_bytes(canonical_json_bytes(report))
        receipt_path.write_bytes(canonical_json_bytes(receipt))
    except OSError as error:
        _fail("output", f"cannot write canonical evidence: {error}")
    return report, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--command-records", required=True, type=Path)
    parser.add_argument("--subject-records", required=True, type=Path)
    parser.add_argument("--gpu-csv", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--build-image-id", required=True)
    parser.add_argument("--profile-binary", required=True, type=Path)
    parser.add_argument("--model-tree-sha256", required=True)
    parser.add_argument("--recorded-at-utc", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report, receipt = write_evidence(
            args.evidence_dir,
            command_records=args.command_records,
            subject_records=args.subject_records,
            gpu_csv=args.gpu_csv,
            report_path=args.report,
            receipt_path=args.receipt,
            source_revision=args.source_revision,
            source_archive_sha256=args.source_archive_sha256,
            build_image_id=args.build_image_id,
            profile_binary=args.profile_binary,
            model_tree_sha256=args.model_tree_sha256,
            recorded_at_utc=args.recorded_at_utc,
        )
    except (EvidenceWriterError, OSError) as error:
        print(f"optimization evidence writer rejected input: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report_sha256": hashlib.sha256(canonical_json_bytes(report)).hexdigest(),
                "receipt_sha256": hashlib.sha256(canonical_json_bytes(receipt)).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
