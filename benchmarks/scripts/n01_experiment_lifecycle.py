#!/usr/bin/env python3
"""Record one N01 benchmark lifecycle without entering the serving runtime.

This is an offline parent-process controller.  It invokes the manifest's
already-built commands through argv arrays and samples ``nvidia-smi`` while
they run; Riley's Rust -> native ABI -> CUDA serving path never imports this
module.  The controller deliberately has no Blender start, stop, or restore
operation.  A failed phase prevents later workload phases from running, but
the manifest's shutdown command is still attempted and a terminal receipt is
always written.
"""

from __future__ import annotations

import argparse
import csv
import datetime as datetime_module
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


NEXT_STAGE_DIRECTORY = Path(__file__).resolve().parents[1] / "next_stage"
if str(NEXT_STAGE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(NEXT_STAGE_DIRECTORY))

from validate_stage_contract import ContractError, validate_documents


MANIFEST_SCHEMA_VERSION = "riley.n01-experiment-manifest.v1"
RECEIPT_SCHEMA_VERSION = "riley.n01-lifecycle-receipt.v1"
PHASES = (
    "asset-preparation",
    "gpu-initialization",
    "warmup",
    "timed-serving",
    "shutdown",
)
SAMPLING_REQUIRED_PHASES = ("warmup", "timed-serving")
EVIDENCE_SCOPES = {"operator-control", "full-model-serving"}
GLOBAL_GPU_PEAK_CEILING_BYTES = 20_000_000_000
UNCERTAINTY_RESERVE_BYTES = 1_000_000_000
# A lifecycle receipt can only pass when the sampled whole-GPU observation is
# at or below this reserve-adjusted bound.  The 20 GB ceiling remains recorded
# separately so receipts retain both the absolute ceiling and the safety gap.
SAMPLED_GLOBAL_GPU_PEAK_LIMIT_BYTES = 19_000_000_000
MIB_BYTES = 1024 * 1024
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
ENVIRONMENT_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
BLENDER_EXECUTABLE_NAMES = {"blender", "blender.exe"}
BLENDER_RESTORE_HELPER_NAMES = {
    "restore-blender",
    "blender-restore",
    "restore_blender",
    "blender_restore",
}
SHELL_BLENDER_COMMAND_PATTERN = re.compile(
    r"(?:^|[;&|()\s])(?:[^\s;&|()]+/)?blender(?:\.exe)?(?=$|[;&|()\s])",
    re.IGNORECASE,
)


class ManifestError(ValueError):
    """The manifest is not a safe, complete N01 lifecycle contract."""


def reject_duplicate_json_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise ManifestError(f"duplicate JSON key {key!r}")
        value[key] = member
    return value


def parse_json_bytes(data: bytes, location: str) -> Any:
    try:
        return json.loads(data, object_pairs_hook=reject_duplicate_json_keys)
    except json.JSONDecodeError as error:
        raise ManifestError(f"{location} is not valid JSON: {error}") from error


def utc_now() -> str:
    return (
        datetime_module.datetime.now(datetime_module.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_exact_keys(value: object, keys: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{location} must be an object")
    found = set(value)
    missing = sorted(keys - found)
    unexpected = sorted(found - keys)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ManifestError(f"{location} has " + "; ".join(details))
    return value


def validate_source_binding(value: object, location: str, include_id: bool = False) -> dict[str, Any]:
    keys = {"path", "sha256"}
    if include_id:
        keys.add("id")
    value = require_exact_keys(value, keys, location)
    if not isinstance(value["path"], str) or not value["path"]:
        raise ManifestError(f"{location}.path must be a nonempty string")
    if not isinstance(value["sha256"], str) or not SHA256_PATTERN.fullmatch(value["sha256"]):
        raise ManifestError(f"{location}.sha256 must be a lowercase SHA-256 digest")
    if include_id and (not isinstance(value["id"], str) or not ID_PATTERN.fullmatch(value["id"])):
        raise ManifestError(f"{location}.id must be a contract identifier")
    return value


def validate_contract_bindings(value: object) -> dict[str, Any]:
    bindings = require_exact_keys(
        value,
        {"model_descriptor", "numerical_profile", "checkpoint_provenance"},
        "manifest.contract_bindings",
    )
    validate_source_binding(bindings["model_descriptor"], "manifest.contract_bindings.model_descriptor")
    validate_source_binding(bindings["numerical_profile"], "manifest.contract_bindings.numerical_profile", include_id=True)
    checkpoint = bindings["checkpoint_provenance"]
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("status"), str):
        raise ManifestError("manifest.contract_bindings.checkpoint_provenance must declare a status")
    if checkpoint["status"] == "materialized":
        checkpoint = require_exact_keys(
            checkpoint,
            {"status", "path", "sha256"},
            "manifest.contract_bindings.checkpoint_provenance",
        )
        if not isinstance(checkpoint["path"], str) or not checkpoint["path"]:
            raise ManifestError("materialized checkpoint provenance path must be a nonempty string")
        if not isinstance(checkpoint["sha256"], str) or not SHA256_PATTERN.fullmatch(checkpoint["sha256"]):
            raise ManifestError("materialized checkpoint provenance sha256 must be a lowercase SHA-256 digest")
    elif checkpoint["status"] == "materialization-pending":
        require_exact_keys(
            checkpoint,
            {"status"},
            "manifest.contract_bindings.checkpoint_provenance",
        )
    else:
        raise ManifestError("checkpoint provenance status must be materialized or materialization-pending")
    return bindings


def validate_target(value: object) -> dict[str, Any]:
    target = require_exact_keys(value, {"model_id", "evidence_scope"}, "manifest.target")
    if not isinstance(target["model_id"], str) or not ID_PATTERN.fullmatch(target["model_id"]):
        raise ManifestError("manifest.target.model_id must be a contract identifier")
    if target["evidence_scope"] not in EVIDENCE_SCOPES:
        raise ManifestError(
            "manifest.target.evidence_scope must be operator-control or full-model-serving"
        )
    return target


def resolved_binding_path(manifest_path: Path, declared_path: str, location: str) -> Path:
    """Resolve one immutable evidence input relative to its manifest.

    Absolute paths make a remote receipt independently reproducible. Relative
    paths are intentionally resolved from the manifest rather than the
    controller's current directory so an invocation cannot bind a different
    file merely because it was launched elsewhere.
    """
    candidate = Path(declared_path)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ManifestError(f"{location}.path cannot be resolved: {declared_path!r}") from error
    if not resolved.is_file():
        raise ManifestError(f"{location}.path must resolve to a regular file: {declared_path!r}")
    return resolved


def verify_file_binding(
    manifest_path: Path,
    binding: dict[str, Any],
    location: str,
) -> tuple[dict[str, str], bytes]:
    """Read an evidence file and prove it matches its declared SHA-256."""
    resolved = resolved_binding_path(manifest_path, binding["path"], location)
    try:
        contents = resolved.read_bytes()
    except OSError as error:
        raise ManifestError(f"{location}.path could not be read: {binding['path']!r}") from error
    observed_sha256 = sha256_bytes(contents)
    if observed_sha256 != binding["sha256"]:
        raise ManifestError(
            f"{location}.sha256 does not match {binding['path']!r}: "
            f"declared {binding['sha256']}, observed {observed_sha256}"
        )
    return (
        {
            "resolved_path": str(resolved),
            "verified_sha256": observed_sha256,
        },
        contents,
    )


def verify_contract_bindings(
    manifest_path: Path,
    bindings: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    """Verify immutable bytes plus the pinned model/profile semantic linkage."""
    model_descriptor, model_descriptor_bytes = verify_file_binding(
        manifest_path,
        bindings["model_descriptor"],
        "manifest.contract_bindings.model_descriptor",
    )
    numerical_profile, numerical_profile_bytes = verify_file_binding(
        manifest_path,
        bindings["numerical_profile"],
        "manifest.contract_bindings.numerical_profile",
    )
    numerical_profile["id"] = bindings["numerical_profile"]["id"]

    models_document = parse_json_bytes(
        model_descriptor_bytes,
        "manifest.contract_bindings.model_descriptor",
    )
    numerical_document = parse_json_bytes(
        numerical_profile_bytes,
        "manifest.contract_bindings.numerical_profile",
    )
    try:
        validate_documents(models_document, numerical_document)
    except ContractError as error:
        raise ManifestError(f"N01 model/numerical contract is invalid: {error}") from error
    models = {model["id"]: model for model in models_document["models"]}
    profiles = {profile["id"]: profile for profile in numerical_document["profiles"]}
    target_model = models.get(target["model_id"])
    if target_model is None:
        raise ManifestError(
            f"manifest.target.model_id {target['model_id']!r} is not present in the verified descriptor"
        )
    target_profile = profiles.get(bindings["numerical_profile"]["id"])
    if target_profile is None:
        raise ManifestError(
            "manifest.contract_bindings.numerical_profile.id is not present in the verified numerical profile"
        )
    if (
        bindings["numerical_profile"]["id"] not in target_model["numerical_profiles"]
        or target["model_id"] not in target_profile["applies_to"]
    ):
        raise ManifestError(
            "manifest target model and numerical profile are not linked in the verified N01 contract"
        )

    checkpoint = bindings["checkpoint_provenance"]
    if checkpoint["status"] == "materialization-pending":
        checkpoint_verification: dict[str, Any] = {"status": "materialization-pending"}
    else:
        checkpoint_verification = {
            "status": "materialized",
            **verify_file_binding(
                manifest_path,
                checkpoint,
                "manifest.contract_bindings.checkpoint_provenance",
            )[0],
        }
    if target["evidence_scope"] == "full-model-serving":
        if checkpoint["status"] != "materialized":
            raise ManifestError("full-model-serving evidence requires a materialized checkpoint provenance")
        if target_model["serving_support"]["status"] != "supported-existing-strict-path":
            raise ManifestError(
                "full-model-serving evidence requires a model marked supported by the verified N01 descriptor"
            )
    return {
        "model_descriptor": model_descriptor,
        "numerical_profile": numerical_profile,
        "checkpoint_provenance": checkpoint_verification,
        "target": {
            "model_id": target["model_id"],
            "numerical_profile_id": bindings["numerical_profile"]["id"],
            "evidence_scope": target["evidence_scope"],
            "semantic_contract": "validated-n01-v1",
        },
    }


def is_forbidden_blender_command(argv: list[str]) -> bool:
    """Reject executable Blender actions without rejecting historical receipt filenames."""
    for token in argv:
        basename = Path(token.strip()).name.casefold()
        if basename in BLENDER_EXECUTABLE_NAMES | BLENDER_RESTORE_HELPER_NAMES:
            return True
        if SHELL_BLENDER_COMMAND_PATTERN.search(token):
            return True
    return False


def validate_phase(phase: object, expected_name: str, index: int) -> dict[str, Any]:
    location = f"phases[{index}]"
    if not isinstance(phase, dict):
        raise ManifestError(f"{location} must be an object")
    allowed = {"name", "argv", "timeout_seconds", "cwd", "environment"}
    required = {"name", "argv", "timeout_seconds"}
    found = set(phase)
    if missing := sorted(required - found):
        raise ManifestError(f"{location} is missing " + ", ".join(missing))
    if unexpected := sorted(found - allowed):
        raise ManifestError(f"{location} has unexpected " + ", ".join(unexpected))
    if phase["name"] != expected_name:
        raise ManifestError(f"{location}.name must be {expected_name!r}")
    argv = phase["argv"]
    if not isinstance(argv, list) or not argv or not all(isinstance(token, str) and token for token in argv):
        raise ManifestError(f"{location}.argv must be a nonempty string array")
    if is_forbidden_blender_command(argv):
        raise ManifestError(f"{location}.argv may not invoke Blender or a Blender restoration helper")
    if not is_int(phase["timeout_seconds"]) or not 1 <= phase["timeout_seconds"] <= 86_400:
        raise ManifestError(f"{location}.timeout_seconds must be an integer between 1 and 86400")
    if "cwd" in phase and (not isinstance(phase["cwd"], str) or not phase["cwd"]):
        raise ManifestError(f"{location}.cwd must be a nonempty string when present")
    if "environment" in phase:
        environment = phase["environment"]
        if not isinstance(environment, dict):
            raise ManifestError(f"{location}.environment must be an object")
        for key, value in environment.items():
            if not isinstance(key, str) or not ENVIRONMENT_KEY_PATTERN.fullmatch(key):
                raise ManifestError(f"{location}.environment has an invalid key")
            if not isinstance(value, str):
                raise ManifestError(f"{location}.environment values must be strings")
    return phase


def resolve_phase_cwd(manifest_path: Path, phase: dict[str, Any]) -> str:
    """Bind each phase's current directory to the manifest, not the caller."""
    declared = phase.get("cwd")
    candidate = manifest_path.parent if declared is None else Path(declared)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ManifestError(f"phase {phase['name']!r} cwd cannot be resolved") from error
    if not resolved.is_dir():
        raise ManifestError(f"phase {phase['name']!r} cwd must resolve to a directory")
    return str(resolved)


def validate_manifest(manifest: object) -> dict[str, Any]:
    manifest = require_exact_keys(
        manifest,
        {
            "schema_version",
            "experiment_id",
            "lifecycle_policy",
            "contract_bindings",
            "target",
            "gpu",
            "memory_budget",
            "phases",
        },
        "manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"manifest.schema_version must be {MANIFEST_SCHEMA_VERSION!r}")
    if not isinstance(manifest["experiment_id"], str) or not ID_PATTERN.fullmatch(manifest["experiment_id"]):
        raise ManifestError("manifest.experiment_id must be a nonempty contract identifier")
    if manifest["lifecycle_policy"] != "leave_stopped_no_restore":
        raise ManifestError("manifest.lifecycle_policy must be leave_stopped_no_restore")
    validate_contract_bindings(manifest["contract_bindings"])
    validate_target(manifest["target"])

    gpu = require_exact_keys(manifest["gpu"], {"nvidia_smi", "gpu_indices", "poll_interval_ms"}, "manifest.gpu")
    if not isinstance(gpu["nvidia_smi"], str) or not gpu["nvidia_smi"]:
        raise ManifestError("manifest.gpu.nvidia_smi must be a nonempty command path")
    gpu_indices = gpu["gpu_indices"]
    if gpu_indices != [0]:
        raise ManifestError(
            "N01 v1 permits only physical GPU index [0]; later multi-GPU work needs per-device budgets"
        )
    if not is_int(gpu["poll_interval_ms"]) or not 10 <= gpu["poll_interval_ms"] <= 60_000:
        raise ManifestError("manifest.gpu.poll_interval_ms must be an integer between 10 and 60000")

    memory_budget = require_exact_keys(
        manifest["memory_budget"],
        {"global_peak_ceiling_bytes", "uncertainty_reserve_bytes"},
        "manifest.memory_budget",
    )
    if memory_budget["global_peak_ceiling_bytes"] != GLOBAL_GPU_PEAK_CEILING_BYTES:
        raise ManifestError(f"manifest.memory_budget.global_peak_ceiling_bytes must be {GLOBAL_GPU_PEAK_CEILING_BYTES}")
    if memory_budget["uncertainty_reserve_bytes"] != UNCERTAINTY_RESERVE_BYTES:
        raise ManifestError(f"manifest.memory_budget.uncertainty_reserve_bytes must be {UNCERTAINTY_RESERVE_BYTES}")

    phases = manifest["phases"]
    if not isinstance(phases, list) or len(phases) != len(PHASES):
        raise ManifestError("manifest.phases must contain the five required lifecycle phases exactly once")
    for index, expected_name in enumerate(PHASES):
        validate_phase(phases[index], expected_name, index)
    return manifest


def parse_memory_mib(value: str) -> int:
    normalized = value.strip().removesuffix("MiB").strip()
    try:
        parsed = Decimal(normalized)
    except InvalidOperation as error:
        raise ValueError(f"not a memory value in MiB: {value!r}") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"not a nonnegative memory value in MiB: {value!r}")
    return int(parsed * MIB_BYTES)


def parse_nvidia_smi(output: str, selected_indices: list[int]) -> list[dict[str, Any]]:
    rows = list(csv.reader(line for line in output.splitlines() if line.strip()))
    if not rows:
        raise ValueError("nvidia-smi returned no GPU rows")
    devices: dict[int, dict[str, Any]] = {}
    for row in rows:
        if len(row) != 5:
            raise ValueError("nvidia-smi row must contain index, uuid, name, memory.used, memory.total")
        try:
            index = int(row[0].strip())
        except ValueError as error:
            raise ValueError(f"invalid nvidia-smi GPU index: {row[0]!r}") from error
        if index in devices:
            raise ValueError(f"duplicate nvidia-smi GPU index: {index}")
        devices[index] = {
            "index": index,
            "uuid": row[1].strip(),
            "name": row[2].strip(),
            "used_bytes": parse_memory_mib(row[3]),
            "total_bytes": parse_memory_mib(row[4]),
        }
    missing = [index for index in selected_indices if index not in devices]
    if missing:
        raise ValueError("nvidia-smi did not report selected GPU index " + ", ".join(map(str, missing)))
    return [devices[index] for index in selected_indices]


def parse_proc_status_rss(status: str) -> int | None:
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) < 2:
                raise ValueError("VmRSS line lacks a value")
            return int(fields[1]) * 1024
    return None


def parse_proc_meminfo(meminfo: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in meminfo.splitlines():
        if ":" not in line:
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        values[fields[0].removesuffix(":")] = int(fields[1]) * 1024
    return values


def linux_host_memory_snapshot() -> dict[str, int | None]:
    """Read controller RSS and host memory when procfs is available.

    ``None`` is deliberate on hosts without Linux procfs; that does not change
    the GPU observation result, but it keeps the absence explicit in the
    receipt rather than substituting a made-up zero.
    """
    controller_rss_bytes: int | None = None
    host_memory_total_bytes: int | None = None
    host_memory_available_bytes: int | None = None
    try:
        controller_rss_bytes = parse_proc_status_rss(Path("/proc/self/status").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    try:
        values = parse_proc_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
        host_memory_total_bytes = values.get("MemTotal")
        host_memory_available_bytes = values.get("MemAvailable")
    except (OSError, ValueError):
        pass
    return {
        "controller_rss_bytes": controller_rss_bytes,
        "host_memory_total_bytes": host_memory_total_bytes,
        "host_memory_available_bytes": host_memory_available_bytes,
    }


class GpuSampler:
    def __init__(self, nvidia_smi: str, selected_indices: list[int]) -> None:
        self._nvidia_smi = nvidia_smi
        self._selected_indices = selected_indices
        self.samples: list[dict[str, Any]] = []

    def sample(self, phase: str | None, event: str) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "sample_index": len(self.samples),
            "at_utc": utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "phase": phase,
            "event": event,
        }
        sample.update(linux_host_memory_snapshot())
        try:
            completed = subprocess.run(
                [
                    self._nvidia_smi,
                    "--query-gpu=index,uuid,name,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"nvidia-smi exit {completed.returncode}: {completed.stderr.strip()}")
            devices = parse_nvidia_smi(completed.stdout, self._selected_indices)
            sample.update(
                ok=True,
                devices=devices,
                global_gpu_used_bytes=sum(device["used_bytes"] for device in devices),
                global_gpu_total_bytes=sum(device["total_bytes"] for device in devices),
                error=None,
            )
        except Exception as error:  # Poll failure is evidence and fails the final gate, not the receipt write.
            sample.update(
                ok=False,
                devices=[],
                global_gpu_used_bytes=None,
                global_gpu_total_bytes=None,
                error=f"{type(error).__name__}: {error}",
            )
        self.samples.append(sample)
        return sample


def host_memory_extrema(samples: list[dict[str, Any]]) -> dict[str, int | None]:
    rss_values = [sample["controller_rss_bytes"] for sample in samples if sample["controller_rss_bytes"] is not None]
    available_values = [
        sample["host_memory_available_bytes"]
        for sample in samples
        if sample["host_memory_available_bytes"] is not None
    ]
    total_values = [sample["host_memory_total_bytes"] for sample in samples if sample["host_memory_total_bytes"] is not None]
    return {
        "controller_rss_peak_bytes": max(rss_values, default=None),
        "host_memory_available_min_bytes": min(available_values, default=None),
        "host_memory_total_bytes": max(total_values, default=None),
    }


def phase_memory(samples: list[dict[str, Any]], start_index: int, end_index: int) -> dict[str, Any]:
    phase_samples = samples[start_index : end_index + 1]
    host_memory = host_memory_extrema(phase_samples)
    valid = [sample for sample in phase_samples if sample["ok"]]
    live_polls = [sample for sample in valid if sample["event"] == "in-process-poll"]
    if not valid:
        return {
            "sample_count": len(phase_samples),
            "successful_sample_count": 0,
            "live_poll_sample_count": 0,
            "baseline_global_gpu_used_bytes": None,
            "peak_global_gpu_used_bytes": None,
            "peak_over_phase_baseline_bytes": None,
            "within_global_peak_ceiling": False,
            "reserve_satisfied": False,
            **host_memory,
        }
    baseline = valid[0]["global_gpu_used_bytes"]
    peak = max(sample["global_gpu_used_bytes"] for sample in valid)
    return {
        "sample_count": len(phase_samples),
        "successful_sample_count": len(valid),
        "live_poll_sample_count": len(live_polls),
        "baseline_global_gpu_used_bytes": baseline,
        "peak_global_gpu_used_bytes": peak,
        "peak_over_phase_baseline_bytes": peak - baseline,
        "within_global_peak_ceiling": peak <= GLOBAL_GPU_PEAK_CEILING_BYTES,
        "reserve_satisfied": peak <= SAMPLED_GLOBAL_GPU_PEAK_LIMIT_BYTES,
        **host_memory,
    }


def command_digest(argv: list[str]) -> str:
    return sha256_bytes(json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def phase_log_name(receipt_path: Path, index: int, name: str, stream: str) -> str:
    return f"{receipt_path.stem}.{index:02d}-{name}.{stream}.log"


def child_environment(phase: dict[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(phase.get("environment", {}))
    return environment


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def skipped_phase(
    phase: dict[str, Any],
    index: int,
    reason: str,
    sampler: GpuSampler,
    cwd: str,
) -> dict[str, Any]:
    name = phase["name"]
    start_index = len(sampler.samples)
    started = utc_now()
    sampler.sample(name, "skipped")
    end_index = len(sampler.samples) - 1
    return {
        "name": name,
        "state": "skipped",
        "argv": phase["argv"],
        "argv_sha256": command_digest(phase["argv"]),
        "cwd": cwd,
        "timeout_seconds": phase["timeout_seconds"],
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "elapsed_ms": 0,
        "exit_code": None,
        "error": reason,
        "stdout_log": None,
        "stderr_log": None,
        "sample_indices": list(range(start_index, end_index + 1)),
        "memory": phase_memory(sampler.samples, start_index, end_index),
    }


def execute_phase(
    phase: dict[str, Any],
    index: int,
    receipt_path: Path,
    sampler: GpuSampler,
    poll_interval_seconds: float,
    cwd: str,
) -> dict[str, Any]:
    name = phase["name"]
    start_index = len(sampler.samples)
    started_at = utc_now()
    started_ns = time.monotonic_ns()
    sampler.sample(name, "phase-start")
    stdout_name = phase_log_name(receipt_path, index, name, "stdout")
    stderr_name = phase_log_name(receipt_path, index, name, "stderr")
    stdout_path = receipt_path.parent / stdout_name
    stderr_path = receipt_path.parent / stderr_name
    process: subprocess.Popen[str] | None = None
    state = "failed"
    error: str | None = None
    exit_code: int | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                phase["argv"],
                cwd=cwd,
                env=child_environment(phase),
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
                text=True,
            )
            if process.poll() is None:
                sampler.sample(name, "post-launch")
            deadline = time.monotonic() + phase["timeout_seconds"]
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    state = "timed-out"
                    error = f"command exceeded {phase['timeout_seconds']} seconds"
                    stop_process(process)
                    break
                time.sleep(poll_interval_seconds)
                if process.poll() is None:
                    sampler.sample(name, "in-process-poll")
            if process.poll() is not None:
                exit_code = process.returncode
            if state != "timed-out":
                if exit_code == 0:
                    state = "completed"
                else:
                    state = "failed"
                    error = f"command exited with code {exit_code}"
    except KeyboardInterrupt:
        state = "interrupted"
        error = "controller interrupted"
        if process is not None:
            stop_process(process)
            exit_code = process.returncode
    except BaseException as exception:  # The receipt must survive controller and command setup failures.
        state = "failed"
        error = f"{type(exception).__name__}: {exception}"
        if process is not None:
            stop_process(process)
            exit_code = process.returncode
    finally:
        sampler.sample(name, "phase-end")
    end_index = len(sampler.samples) - 1
    return {
        "name": name,
        "state": state,
        "argv": phase["argv"],
        "argv_sha256": command_digest(phase["argv"]),
        "cwd": cwd,
        "timeout_seconds": phase["timeout_seconds"],
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "elapsed_ms": (time.monotonic_ns() - started_ns) // 1_000_000,
        "exit_code": exit_code,
        "error": error,
        "stdout_log": stdout_name,
        "stderr_log": stderr_name,
        "sample_indices": list(range(start_index, end_index + 1)),
        "memory": phase_memory(sampler.samples, start_index, end_index),
    }


def controller_failure_phase(
    phase: dict[str, Any],
    error: str,
    cwd: str,
) -> dict[str, Any]:
    """Produce a schema-complete phase record if the controller itself failed."""
    now = utc_now()
    return {
        "name": phase["name"],
        "state": "failed",
        "argv": phase["argv"],
        "argv_sha256": command_digest(phase["argv"]),
        "cwd": cwd,
        "timeout_seconds": phase["timeout_seconds"],
        "started_at_utc": now,
        "finished_at_utc": now,
        "elapsed_ms": 0,
        "exit_code": None,
        "error": error,
        "stdout_log": None,
        "stderr_log": None,
        "sample_indices": [],
        "memory": {
            "sample_count": 0,
            "successful_sample_count": 0,
            "live_poll_sample_count": 0,
            "baseline_global_gpu_used_bytes": None,
            "peak_global_gpu_used_bytes": None,
            "peak_over_phase_baseline_bytes": None,
            "within_global_peak_ceiling": False,
            "reserve_satisfied": False,
            "controller_rss_peak_bytes": None,
            "host_memory_available_min_bytes": None,
            "host_memory_total_bytes": None,
        },
    }


def overall_memory(
    samples: list[dict[str, Any]],
    baseline: dict[str, Any],
    phases: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [sample for sample in samples if sample["ok"]]
    peak = max((sample["global_gpu_used_bytes"] for sample in valid), default=None)
    baseline_used = baseline["global_gpu_used_bytes"] if baseline.get("ok") else None
    observed_headroom = None if peak is None else GLOBAL_GPU_PEAK_CEILING_BYTES - peak
    by_name = {phase["name"]: phase for phase in phases}
    missing_live_samples = [
        name
        for name in SAMPLING_REQUIRED_PHASES
        if name not in by_name or by_name[name]["memory"]["live_poll_sample_count"] == 0
    ]
    return {
        "successful_sample_count": len(valid),
        "failed_sample_count": len(samples) - len(valid),
        "overall_global_gpu_peak_bytes": peak,
        "peak_over_lifecycle_baseline_bytes": None if peak is None or baseline_used is None else peak - baseline_used,
        "headroom_to_global_peak_ceiling_bytes": observed_headroom,
        "within_global_peak_ceiling": peak is not None and peak <= GLOBAL_GPU_PEAK_CEILING_BYTES,
        "reserve_satisfied": peak is not None and peak <= SAMPLED_GLOBAL_GPU_PEAK_LIMIT_BYTES,
        "required_in_process_phase_count": len(SAMPLING_REQUIRED_PHASES),
        "phases_with_in_process_samples": len(SAMPLING_REQUIRED_PHASES) - len(missing_live_samples),
        "missing_in_process_sample_phases": missing_live_samples,
        "sampled_peak_evidence_status": (
            "sampled-observed-not-continuous"
            if not missing_live_samples
            else "insufficient-in-process-sampling"
        ),
        **host_memory_extrema(samples),
    }


def invalid_manifest_receipt(manifest_path: Path, manifest_bytes: bytes | None, error: Exception) -> dict[str, Any]:
    now = utc_now()
    phases = []
    for name in PHASES:
        phases.append(
            {
                "name": name,
                "state": "skipped",
                "argv": [],
                "argv_sha256": None,
                "cwd": None,
                "timeout_seconds": None,
                "started_at_utc": now,
                "finished_at_utc": now,
                "elapsed_ms": 0,
                "exit_code": None,
                "error": "manifest validation failed before phase execution",
                "stdout_log": None,
                "stderr_log": None,
                "sample_indices": [],
                "memory": {
                    "sample_count": 0,
                    "successful_sample_count": 0,
                    "live_poll_sample_count": 0,
                    "baseline_global_gpu_used_bytes": None,
                    "peak_global_gpu_used_bytes": None,
                    "peak_over_phase_baseline_bytes": None,
                    "within_global_peak_ceiling": False,
                    "reserve_satisfied": False,
                    "controller_rss_peak_bytes": None,
                    "host_memory_available_min_bytes": None,
                    "host_memory_total_bytes": None,
                },
            }
        )
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "terminal": True,
        "status": "invalid-manifest",
        "experiment_id": None,
        "manifest": {
            "path": str(manifest_path),
            "sha256": None if manifest_bytes is None else sha256_bytes(manifest_bytes),
            "validation_error": f"{type(error).__name__}: {error}",
        },
        "contract_bindings": None,
        "contract_binding_verification": None,
        "lifecycle_policy": "leave_stopped_no_restore",
        "phase_order": list(PHASES),
        "memory_budget": {
            "global_peak_ceiling_bytes": GLOBAL_GPU_PEAK_CEILING_BYTES,
            "uncertainty_reserve_bytes": UNCERTAINTY_RESERVE_BYTES,
            "reserve_adjusted_limit_bytes": SAMPLED_GLOBAL_GPU_PEAK_LIMIT_BYTES,
        },
        "baseline": None,
        "phases": phases,
        "samples": [],
        "overall": {
            "successful_sample_count": 0,
            "failed_sample_count": 0,
            "overall_global_gpu_peak_bytes": None,
            "peak_over_lifecycle_baseline_bytes": None,
            "headroom_to_global_peak_ceiling_bytes": None,
            "within_global_peak_ceiling": False,
            "reserve_satisfied": False,
            "required_in_process_phase_count": len(SAMPLING_REQUIRED_PHASES),
            "phases_with_in_process_samples": 0,
            "missing_in_process_sample_phases": list(SAMPLING_REQUIRED_PHASES),
            "sampled_peak_evidence_status": "insufficient-in-process-sampling",
            "controller_rss_peak_bytes": None,
            "host_memory_available_min_bytes": None,
            "host_memory_total_bytes": None,
        },
        "failure_reasons": [f"manifest validation failed: {type(error).__name__}: {error}"],
        "finished_at_utc": now,
    }


def run_manifest(manifest_path: Path, receipt_path: Path) -> int:
    manifest_bytes: bytes | None = None
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = validate_manifest(parse_json_bytes(manifest_bytes, "manifest"))
        phase_cwds = [resolve_phase_cwd(manifest_path, phase) for phase in manifest["phases"]]
        binding_verification = verify_contract_bindings(
            manifest_path,
            manifest["contract_bindings"],
            manifest["target"],
        )
    except Exception as error:
        write_json(receipt_path, invalid_manifest_receipt(manifest_path, manifest_bytes, error))
        return 2

    sampler = GpuSampler(manifest["gpu"]["nvidia_smi"], manifest["gpu"]["gpu_indices"])
    phase_records: dict[str, dict[str, Any]] = {}
    failure_reasons: list[str] = []
    baseline = sampler.sample(None, "lifecycle-baseline")
    if not baseline["ok"]:
        failure_reasons.append("lifecycle baseline GPU sample failed: " + baseline["error"])

    predecessor_failed = not baseline["ok"]
    try:
        for index, phase in enumerate(manifest["phases"][:-1]):
            if predecessor_failed:
                record = skipped_phase(
                    phase,
                    index,
                    "skipped after a prior lifecycle failure",
                    sampler,
                    phase_cwds[index],
                )
            else:
                record = execute_phase(
                    phase,
                    index,
                    receipt_path,
                    sampler,
                    manifest["gpu"]["poll_interval_ms"] / 1000,
                    phase_cwds[index],
                )
            phase_records[phase["name"]] = record
            if record["state"] != "completed":
                predecessor_failed = True
                failure_reasons.append(f"{record['name']} {record['state']}: {record['error']}")
    except BaseException as error:  # Keep the terminal receipt even for an unexpected controller failure.
        failure_reasons.append(f"controller failure: {type(error).__name__}: {error}")
    finally:
        shutdown_index = len(PHASES) - 1
        shutdown_phase = manifest["phases"][shutdown_index]
        if shutdown_phase["name"] not in phase_records:
            try:
                shutdown = execute_phase(
                    shutdown_phase,
                    shutdown_index,
                    receipt_path,
                    sampler,
                    manifest["gpu"]["poll_interval_ms"] / 1000,
                    phase_cwds[shutdown_index],
                )
            except BaseException as error:
                shutdown = controller_failure_phase(
                    shutdown_phase,
                    f"controller failure while attempting shutdown: {type(error).__name__}: {error}",
                    phase_cwds[shutdown_index],
                )
            phase_records[shutdown_phase["name"]] = shutdown
            if shutdown["state"] != "completed":
                failure_reasons.append(f"shutdown {shutdown['state']}: {shutdown['error']}")

        for index, phase in enumerate(manifest["phases"][:-1]):
            if phase["name"] in phase_records:
                continue
            try:
                phase_records[phase["name"]] = skipped_phase(
                    phase,
                    index,
                    "controller failure before phase execution",
                    sampler,
                    phase_cwds[index],
                )
            except BaseException as error:
                phase_records[phase["name"]] = controller_failure_phase(
                    phase,
                    f"controller failure while recording skipped phase: {type(error).__name__}: {error}",
                    phase_cwds[index],
                )

    phases = [phase_records[phase["name"]] for phase in manifest["phases"]]
    overall = overall_memory(sampler.samples, baseline, phases)
    if overall["failed_sample_count"]:
        failure_reasons.append(f"{overall['failed_sample_count']} GPU samples failed")
    if not overall["within_global_peak_ceiling"]:
        failure_reasons.append("global GPU peak exceeded the 20,000,000,000-byte ceiling or was unavailable")
    if not overall["reserve_satisfied"]:
        failure_reasons.append(
            "sampled whole-GPU peak exceeded the 19,000,000,000-byte reserve-adjusted limit"
        )
    if overall["missing_in_process_sample_phases"]:
        failure_reasons.append(
            "sampled GPU peak lacks an in-process poll for "
            + ", ".join(overall["missing_in_process_sample_phases"])
        )
    status = "passed" if not failure_reasons else "failed"
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "terminal": True,
        "status": status,
        "experiment_id": manifest["experiment_id"],
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_bytes(manifest_bytes),
            "validation_error": None,
        },
        "contract_bindings": manifest["contract_bindings"],
        "contract_binding_verification": binding_verification,
        "lifecycle_policy": "leave_stopped_no_restore",
        "phase_order": list(PHASES),
        "memory_budget": {
            "global_peak_ceiling_bytes": GLOBAL_GPU_PEAK_CEILING_BYTES,
            "uncertainty_reserve_bytes": UNCERTAINTY_RESERVE_BYTES,
            "reserve_adjusted_limit_bytes": SAMPLED_GLOBAL_GPU_PEAK_LIMIT_BYTES,
        },
        "baseline": baseline,
        "phases": phases,
        "samples": sampler.samples,
        "overall": overall,
        "failure_reasons": failure_reasons,
        "finished_at_utc": utc_now(),
    }
    write_json(receipt_path, receipt)
    return 0 if status == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="N01 experiment manifest JSON")
    parser.add_argument("receipt", type=Path, help="terminal N01 lifecycle receipt JSON")
    arguments = parser.parse_args(argv)
    return run_manifest(arguments.manifest, arguments.receipt)


if __name__ == "__main__":
    sys.exit(main())
