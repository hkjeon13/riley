#!/usr/bin/env python3
"""Run one bounded, paired N06-A Qwen2.5-3B D128 serving attempt.

This is an *external* benchmark controller.  It never participates in the
Rust serving process: it starts one localhost Riley process, stops it, then
starts one localhost vLLM process and stops it (or the reverse on even timed
attempts).  The parent ``n01_repeat_control.py`` injects the attempt phase and
one-based index into this program's child environment.  Timed attempts use
that index to alternate the engine order and emit the two strict marker lines
consumed by ``n03b_n06a_d128_repeat_summary.py``.

The workload is a small, immutable JSON document containing the exact prompt
text, returned prompt IDs, expected greedy output IDs/text, and fixed sampling
parameters.  Every request is strict: one generated token ID must arrive in
each SSE event, terminal usage must agree with the workload, and both engines
must produce the same specified sequence.  A failed, incomplete, grouped, or
mismatched lane writes its raw records and exits non-zero without completing a
paired serving claim.

All artifacts are create-only below ``--artifact-root``.  The outer N01
controller retains this program's stdout/stderr and host PSI; this program
retains lane startup logs, the Riley stderr snapshot, raw request rows, lane
measurements, and cleanup receipts.  It avoids recursive model hashing so a
shared serving host does not pay avoidable checkpoint I/O for every repeat.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol, TextIO

import serving_token_client_v2 as token_client


WORKLOAD_SCHEMA_VERSION = "riley.n06a-d128-serving-workload.v1"
ARTIFACT_SCHEMA_VERSION = "riley.n06a-paired-serving-attempt.v1"
MARKER_PREFIX = "riley-n06a-d128-serving"
STARTUP_RECEIPT_PREFIX = "RILEY_DECODE_ATTENTION"
REQUESTED_BACKEND_CLI_ID = "native-bf16-paged-split-gqa-d128-two-stage"
RESOLVED_BACKEND_ID = (
    "riley.cuda.ragged-paged-attention.native-bf16-paged-split-gqa."
    "qwen2.5-3b.d128.qh16.kvh2.block16.transition-v2"
)
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL_IDENTITY_MANIFEST_SCHEMA_VERSION = "riley.n06a-model-identity-manifest.v1"
MODEL_IDENTITY_VALIDATION_SCHEMA_VERSION = "riley.n06a-model-identity-validation.v1"
VLLM_PREFIX_CACHING_DISABLE_FLAG = "--no-enable-prefix-caching"
VLLM_GPU_MEMORY_UTILIZATION_FLAG = "--gpu-memory-utilization"
VLLM_GPU_MEMORY_UTILIZATION = "0.65"
WHOLE_GPU_SAMPLED_PEAK_LIMIT_BYTES = 19_000_000_000
VLLM_CHUNKED_PREFILL_ENABLE_FLAG = "--enable-chunked-prefill"
VLLM_KV_CACHE_DTYPE_FLAG = "--kv-cache-dtype"
VLLM_KV_CACHE_DTYPE = "bfloat16"
VLLM_DTYPE_FLAG = "--dtype"
VLLM_DTYPE = "bfloat16"
VLLM_AUTO_BACKEND_REQUESTED = "vllm-auto"
VLLM_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
VLLM_BACKEND_RECEIPT_REGEX_PATTERN = (
    r"Using (?P<backend_resolved>[A-Z0-9_]+) attention backend out of potential backends:"
)

N01_PHASE_ENV = "N01_REPEAT_CONTROL_PHASE"
N01_INDEX_ENV = "N01_REPEAT_CONTROL_INDEX"

MAX_WORKLOAD_BYTES = 4 * 1024 * 1024
MAX_MODEL_IDENTITY_MANIFEST_BYTES = 1 * 1024 * 1024
MAX_MODEL_IDENTITY_METADATA_BYTES = 4 * 1024 * 1024
MAX_MODEL_IDENTITY_GIT_OUTPUT_BYTES = 1 * 1024 * 1024
MODEL_IDENTITY_GIT_TIMEOUT_SECONDS = 30.0
MAX_STARTUP_LOG_BYTES = 8 * 1024 * 1024
MAX_EXECUTABLE_BYTES = 1_024 * 1024 * 1024
# Retained raw rows are intentionally bounded before their hashes are promoted
# into a marker.  The normal N06-A workload is far smaller; this stops a broken
# transport from turning a repeat into an unreviewable artifact tree.
MAX_RETAINED_ROWS_BYTES = 128 * 1024 * 1024
MAX_PROMPT_TOKENS = 32_768
MAX_OUTPUT_TOKENS = 1_024
MAX_CONCURRENCY = 32
MAX_REQUESTS = 10_000
MAX_TIMEOUT_SECONDS = 1_800.0
# ``serving_token_client_v2`` owns one total-deadline watchdog per request and
# deliberately rejects a larger deadline.  Reject it here during configuration
# instead of allowing every request in an otherwise valid lane to fail later.
MAX_TOKEN_CLIENT_REQUEST_TIMEOUT_SECONDS = 120.0
GPU_MEMORY_SAMPLE_INTERVAL_SECONDS = 0.25
MIN_GPU_MEMORY_SAMPLING_INTERVAL_SECONDS = 0.25
MAX_GPU_MEMORY_SAMPLING_INTERVAL_SECONDS = 0.5
# A sampled high-water receipt is useful only if every query returns promptly
# and the observed polling sequence has no silent multi-second hole.  These
# are evidence-quality limits, not serving latency limits: an overloaded host
# that cannot meet them retains the failed receipt but cannot promote a lane.
MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS = 0.5
GPU_MEMORY_SAMPLE_SCHEDULING_SLACK_SECONDS = 0.5
GPU_MEMORY_SAMPLE_COMMAND_TIMEOUT_SECONDS = 5.0
MAX_GPU_MEMORY_SAMPLE_OUTPUT_BYTES = 16 * 1024
GPU_IDLE_CENSUS_MAX_USED_BYTES = 512 * 1024 * 1024
GPU_IDLE_CENSUS_COMMAND_TIMEOUT_SECONDS = 5.0
MAX_DOCKER_CLEANUP_OUTPUT_BYTES = 64 * 1024
# Docker control-plane actions are only cleanup evidence for an already-owned
# lane.  Keep each one bounded independently so a sick daemon cannot turn a
# failed serving attempt into an unbounded controller process.
MAX_DOCKER_CLEANUP_COMMAND_SECONDS = 30.0
DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION = "riley.n06a-docker-container-cleanup.v1"
GPU_IDLE_CENSUS_SCHEMA_VERSION = "riley.n06a-post-lane-gpu-idle-census.v1"
CASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DOCKER_CONTAINER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PHASES = frozenset({"warmup", "timed"})
ALLOWED_TEMPLATE_TOKENS = frozenset(
    {
        "{port}",
        "{model_path}",
        "{model_id}",
        "{concurrency}",
        "{batch_token_budget}",
        "{max_model_len}",
        "{max_output_tokens}",
        "{kv_blocks}",
    }
)


class DriverError(ValueError):
    """The attempt cannot establish a complete paired serving observation."""


class ProcessLike(Protocol):
    pid: int
    returncode: int | None
    stdin: Any

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass(frozen=True)
class AttemptContext:
    phase: str
    index: int

    @property
    def pair_order(self) -> str:
        if self.phase != "timed":
            raise DriverError("only timed attempts have a paired marker order")
        return "riley-vllm" if self.index % 2 else "vllm-riley"


@dataclass(frozen=True)
class Workload:
    source_path: Path
    source_sha256: str
    case: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    output_text: str
    finish_reason: str
    temperature: float
    top_p: float

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def max_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def reference(self) -> token_client.TokenReference:
        return token_client.TokenReference(
            MODEL_ID,
            self.prompt_token_ids,
            self.output_token_ids,
            self.output_text,
            self.finish_reason,
        )


@dataclass(frozen=True)
class ModelIdentityMetadataFile:
    """A small model file that is fully hashed for each outer attempt."""

    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelIdentityShard:
    """A large checkpoint shard identified by its source LFS OID and size.

    The driver deliberately does not rehash multi-gigabyte weights on the
    shared serving host.  The immutable manifest binds the expected LFS OID;
    each attempt proves the local regular file remains at the recorded size.
    """

    relative_path: str
    size_bytes: int
    lfs_oid_sha256: str


@dataclass(frozen=True)
class ModelIdentityManifest:
    source_path: Path
    source_sha256: str
    model_path: Path
    metadata_files: tuple[ModelIdentityMetadataFile, ...]
    shards: tuple[ModelIdentityShard, ...]


@dataclass(frozen=True)
class DriverConfig:
    artifact_root: Path
    workload: Workload
    model_path: Path
    model_identity: ModelIdentityManifest
    model_git: Path
    model_git_sha256: str
    riley_binary: Path
    vllm_template: tuple[str, ...]
    vllm_command_path: Path
    vllm_command_sha256: str
    riley_binary_sha256: str
    vllm_launcher_sha256: str
    vllm_backend_requested: str
    vllm_backend_receipt_regex: re.Pattern[str]
    vllm_startup_required_fragments: tuple[str, ...]
    vllm_image_digest: str
    vllm_gpu_memory_utilization: str
    whole_gpu_sampled_peak_limit_bytes: int
    nvidia_smi: Path
    gpu_memory_sampling_interval_seconds: float
    riley_port: int
    vllm_port: int
    concurrency: int
    batch_token_budget: int
    retained_requests: int
    warmup_requests: int
    max_model_len: int
    kv_blocks: int
    startup_timeout_seconds: float
    request_timeout_seconds: float
    shutdown_timeout_seconds: float
    cuda_visible_devices: str
    vllm_version_template: tuple[str, ...] | None


@dataclass
class ManagedProcess:
    lane: str
    process: ProcessLike
    stdout_path: Path
    stderr_path: Path
    stdout_stream: TextIO | Any
    stderr_stream: TextIO | Any
    started_ns: int
    command: tuple[str, ...]
    used_shutdown_stdin: bool


class GpuMemorySamplerLike(Protocol):
    """Owned whole-GPU sampler used only by this external benchmark driver."""

    def start(self) -> None: ...

    def stop(self) -> Mapping[str, Any]: ...


class WholeGpuMemorySampler:
    """Sample physical GPU-0 memory while one owned serving lane is alive."""

    def __init__(
        self,
        *,
        nvidia_smi: Path,
        interval_seconds: float,
        monotonic_ns: Callable[[], int],
    ) -> None:
        self.nvidia_smi = nvidia_smi
        self.interval_seconds = interval_seconds
        self.monotonic_ns = monotonic_ns
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, int]] = []
        self._errors: list[str] = []

    def _sample_once(self) -> None:
        started_ns = self.monotonic_ns()
        try:
            completed = subprocess.run(
                [
                    str(self.nvidia_smi),
                    "--id=0",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                timeout=GPU_MEMORY_SAMPLE_COMMAND_TIMEOUT_SECONDS,
                shell=False,
                check=False,
            )
            if completed.returncode != 0:
                raise DriverError(f"nvidia-smi exited {completed.returncode}")
            if len(completed.stdout) > MAX_GPU_MEMORY_SAMPLE_OUTPUT_BYTES:
                raise DriverError("nvidia-smi memory sample exceeds 16KiB")
            try:
                text = completed.stdout.decode("utf-8")
            except UnicodeDecodeError as error:
                raise DriverError("nvidia-smi memory sample is not UTF-8") from error
            rows = [line.strip() for line in text.splitlines() if line.strip()]
            if len(rows) != 1 or not re.fullmatch(r"[0-9]+(?:\s+MiB)?", rows[0]):
                raise DriverError("nvidia-smi must return one numeric GPU-0 memory.used MiB row")
            used_bytes = int(rows[0].split()[0], 10) * 1024 * 1024
            finished_ns = self.monotonic_ns()
            if finished_ns < started_ns:
                raise DriverError("whole-GPU sampler monotonic clock moved backwards")
            if finished_ns - started_ns > int(MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS * 1_000_000_000):
                raise DriverError(
                    "nvidia-smi whole-GPU memory sample exceeded "
                    f"{MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS:g}s evidence-quality bound"
                )
            with self._lock:
                self._samples.append(
                    {
                        "started_ns": started_ns,
                        "finished_ns": finished_ns,
                        # Keep the established point-observation field for
                        # downstream readers while binding the query interval.
                        "monotonic_ns": finished_ns,
                        "memory_used_bytes": used_bytes,
                    }
                )
        except BaseException as error:
            with self._lock:
                self._errors.append(f"{type(error).__name__}: {error}")

    def _run(self) -> None:
        interval_ns = int(self.interval_seconds * 1_000_000_000)
        deadline_ns = self.monotonic_ns() + interval_ns
        while True:
            remaining_seconds = max(0.0, (deadline_ns - self.monotonic_ns()) / 1_000_000_000.0)
            if self._stop.wait(remaining_seconds):
                return
            self._sample_once()
            # Advance from the planned cadence rather than from completion so
            # a delayed nvidia-smi call becomes observable in the raw timing.
            deadline_ns += interval_ns

    def start(self) -> None:
        if self._thread is not None:
            raise DriverError("whole-GPU sampler cannot be started twice")
        self._sample_once()
        self._thread = threading.Thread(target=self._run, name="n06a-gpu-memory", daemon=True)
        self._thread.start()

    def stop(self) -> Mapping[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=GPU_MEMORY_SAMPLE_COMMAND_TIMEOUT_SECONDS + self.interval_seconds + 1.0)
            if self._thread.is_alive():
                with self._lock:
                    self._errors.append("sampler thread did not stop within its bounded timeout")
        self._sample_once()
        with self._lock:
            samples = [dict(sample) for sample in self._samples]
            errors = list(self._errors)
        peak = max((sample["memory_used_bytes"] for sample in samples), default=None)
        return {
            "schema_version": "riley.n06a-whole-gpu-memory-samples.v1",
            "gpu_index": 0,
            "sampling_interval_seconds": self.interval_seconds,
            "max_sample_duration_seconds": MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS,
            "max_start_gap_seconds": self.interval_seconds + GPU_MEMORY_SAMPLE_SCHEDULING_SLACK_SECONDS,
            "samples": samples,
            "sample_count": len(samples),
            "peak_used_bytes": peak,
            "errors": errors,
        }


def _gpu_idle_census_command(
    *,
    nvidia_smi: Path,
    argv_suffix: Sequence[str],
    timeout_seconds: float,
    monotonic_ns: Callable[[], int],
) -> dict[str, Any]:
    """Run one bounded, no-shell GPU-idle observation with raw provenance."""
    argv = [str(nvidia_smi), *argv_suffix]
    started_ns = monotonic_ns()
    receipt: dict[str, Any] = {
        "argv": argv,
        "started_ns": started_ns,
        "timeout_seconds": timeout_seconds,
    }
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout_seconds,
            shell=False,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        if len(stdout) > MAX_GPU_MEMORY_SAMPLE_OUTPUT_BYTES or len(stderr) > MAX_GPU_MEMORY_SAMPLE_OUTPUT_BYTES:
            raise DriverError("post-lane nvidia-smi census output exceeds 16KiB")
        receipt.update(
            returncode=completed.returncode,
            stdout_text=stdout.decode("utf-8"),
            stderr_text=stderr.decode("utf-8"),
        )
    except BaseException as error:
        receipt.update(returncode=None, stdout_text="", stderr_text="", error=f"{type(error).__name__}: {error}")
    finished_ns = monotonic_ns()
    receipt["finished_ns"] = finished_ns
    for stream in ("stdout", "stderr"):
        value = receipt[f"{stream}_text"]
        assert isinstance(value, str)
        encoded = value.encode("utf-8")
        receipt[f"{stream}_bytes"] = len(encoded)
        receipt[f"{stream}_sha256"] = _sha256(encoded)
    return receipt


def _default_gpu_idle_census(
    *, nvidia_smi: Path, timeout_seconds: float, monotonic_ns: Callable[[], int]
) -> Mapping[str, Any]:
    """Observe GPU-0 after an owned lane without terminating foreign work."""
    process_suffix = (
        "--id=0",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    )
    memory_suffix = (
        "--id=0",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    )
    process = _gpu_idle_census_command(
        nvidia_smi=nvidia_smi,
        argv_suffix=process_suffix,
        timeout_seconds=timeout_seconds,
        monotonic_ns=monotonic_ns,
    )
    memory = _gpu_idle_census_command(
        nvidia_smi=nvidia_smi,
        argv_suffix=memory_suffix,
        timeout_seconds=timeout_seconds,
        monotonic_ns=monotonic_ns,
    )
    receipt: dict[str, Any] = {
        "schema_version": GPU_IDLE_CENSUS_SCHEMA_VERSION,
        "gpu_index": 0,
        "max_idle_memory_bytes": GPU_IDLE_CENSUS_MAX_USED_BYTES,
        "commands": [process, memory],
        "compute_process_pids": [],
        "memory_used_bytes": None,
        "compliant": False,
        "errors": [],
    }
    if process.get("error") is not None or memory.get("error") is not None:
        receipt["errors"].append("post-lane nvidia-smi census command failed")
        return receipt
    if process.get("returncode") != 0 or memory.get("returncode") != 0:
        receipt["errors"].append("post-lane nvidia-smi census returned non-zero")
        return receipt
    process_text = process["stdout_text"]
    memory_text = memory["stdout_text"]
    assert isinstance(process_text, str) and isinstance(memory_text, str)
    process_rows = [line.strip() for line in process_text.splitlines() if line.strip()]
    if not all(re.fullmatch(r"[1-9][0-9]*", line) for line in process_rows):
        receipt["errors"].append("post-lane compute-process census is malformed")
        return receipt
    receipt["compute_process_pids"] = [int(line, 10) for line in process_rows]
    memory_rows = [line.strip() for line in memory_text.splitlines() if line.strip()]
    if len(memory_rows) != 1 or not re.fullmatch(r"[0-9]+(?:\s+MiB)?", memory_rows[0]):
        receipt["errors"].append("post-lane GPU memory census is malformed")
        return receipt
    receipt["memory_used_bytes"] = int(memory_rows[0].split()[0], 10) * 1024 * 1024
    if receipt["compute_process_pids"]:
        receipt["errors"].append("GPU-0 still has compute processes after owned lane cleanup")
    if receipt["memory_used_bytes"] > GPU_IDLE_CENSUS_MAX_USED_BYTES:
        receipt["errors"].append("GPU-0 idle memory exceeds the 512MiB post-lane bound")
    receipt["compliant"] = not receipt["errors"]
    return receipt


def _validate_post_lane_gpu_idle_census(
    receipt: Mapping[str, Any], *, nvidia_smi: Path, label: str
) -> None:
    """Fail closed before the next lane if cleanup left GPU-0 biased or busy."""
    expected = {
        "schema_version": GPU_IDLE_CENSUS_SCHEMA_VERSION,
        "gpu_index": 0,
        "max_idle_memory_bytes": GPU_IDLE_CENSUS_MAX_USED_BYTES,
        "compute_process_pids": [],
        "compliant": True,
        "errors": [],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise DriverError(f"{label} post-lane GPU idle census differs at {key!r}")
    memory_used_bytes = receipt.get("memory_used_bytes")
    if (
        isinstance(memory_used_bytes, bool)
        or not isinstance(memory_used_bytes, int)
        or not 0 <= memory_used_bytes <= GPU_IDLE_CENSUS_MAX_USED_BYTES
    ):
        raise DriverError(f"{label} post-lane GPU idle census memory bound is invalid")
    commands = receipt.get("commands")
    if not isinstance(commands, list) or len(commands) != 2:
        raise DriverError(f"{label} post-lane GPU idle census lacks two bounded commands")
    expected_suffixes = (
        ["--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        ["--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    )
    for index, (command, suffix) in enumerate(zip(commands, expected_suffixes, strict=True)):
        if not isinstance(command, Mapping):
            raise DriverError(f"{label} post-lane GPU idle command[{index}] is invalid")
        if command.get("argv") != [str(nvidia_smi), *suffix]:
            raise DriverError(f"{label} post-lane GPU idle command[{index}] argv differs")
        if command.get("returncode") != 0 or "error" in command:
            raise DriverError(f"{label} post-lane GPU idle command[{index}] did not succeed")
        timeout = command.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < float(timeout) <= GPU_IDLE_CENSUS_COMMAND_TIMEOUT_SECONDS:
            raise DriverError(f"{label} post-lane GPU idle command[{index}] timeout is invalid")
        started_ns = command.get("started_ns")
        finished_ns = command.get("finished_ns")
        if (
            isinstance(started_ns, bool)
            or not isinstance(started_ns, int)
            or isinstance(finished_ns, bool)
            or not isinstance(finished_ns, int)
            or finished_ns < started_ns
        ):
            raise DriverError(f"{label} post-lane GPU idle command[{index}] timing is invalid")
        for stream in ("stdout", "stderr"):
            text = command.get(f"{stream}_text")
            if not isinstance(text, str):
                raise DriverError(f"{label} post-lane GPU idle command[{index}] {stream} is invalid")
            raw = text.encode("utf-8")
            if command.get(f"{stream}_bytes") != len(raw) or command.get(f"{stream}_sha256") != _sha256(raw):
                raise DriverError(f"{label} post-lane GPU idle command[{index}] {stream} receipt differs")


@dataclass(frozen=True)
class StartupSnapshot:
    path: Path
    sha256: str
    receipt: Mapping[str, str]


@dataclass(frozen=True)
class VllmStartupSnapshot:
    stdout_path: Path
    stdout_sha256: str
    stderr_path: Path
    stderr_sha256: str
    backend_requested: str
    backend_resolved: str


@dataclass(frozen=True)
class LaneMeasurement:
    lane: str
    request_rows: tuple[Mapping[str, Any], ...]
    accounting: Mapping[str, Any]
    retained_wall_ms: float
    output_tokens_per_second: float
    ttft: Mapping[str, float]
    tpot: Mapping[str, float]
    e2e: Mapping[str, float]
    p99_status: str


@dataclass(frozen=True)
class RetainedArtifacts:
    """Hashes that bind a compact marker to its create-only retained evidence."""

    request_rows_path: Path
    request_rows_sha256: str
    phase_path: Path
    phase_sha256: str
    attempt_config_path: Path
    attempt_config_sha256: str


@dataclass(frozen=True)
class ServerWarmupArtifacts:
    """Hashes for the strict per-server warmup that precedes retained rows."""

    request_rows_path: Path
    request_rows_sha256: str
    phase_path: Path
    phase_sha256: str


@dataclass(frozen=True)
class LaneProvenanceArtifact:
    """Immutable per-lane launch, cleanup, snapshot, and sampler receipt."""

    path: Path
    sha256: str


@dataclass
class DriverDependencies:
    """Injectable seams keep unit tests independent of CUDA, HTTP, and vLLM."""

    popen: Callable[..., ProcessLike] = subprocess.Popen
    wait_ready: Callable[[ProcessLike, int, str, float], Mapping[str, Any]] | None = None
    run_streaming_phase: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] | None = None
    stop_process: Callable[[ManagedProcess, float], Mapping[str, Any]] | None = None
    cleanup_vllm_container: Callable[[str, str, float], Mapping[str, Any]] | None = None
    gpu_memory_sampler_factory: Callable[..., GpuMemorySamplerLike] | None = None
    gpu_idle_census: Callable[..., Mapping[str, Any]] | None = None
    model_identity_validator: Callable[..., Mapping[str, Any]] | None = None
    monotonic_ns: Callable[[], int] = time.perf_counter_ns
    sleep: Callable[[float], None] = time.sleep


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, maximum_bytes: int, label: str) -> tuple[Path, bytes, str]:
    try:
        before = path.lstat()
    except OSError as error:
        raise DriverError(f"{label} cannot be inspected: {path}: {error}") from error
    if stat.S_ISLNK(before.st_mode):
        raise DriverError(f"{label} must not be a symbolic link: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise DriverError(f"{label} must be a regular file: {path}")
    if before.st_size > maximum_bytes:
        raise DriverError(f"{label} exceeds {maximum_bytes} bytes: {path}")
    try:
        payload = path.read_bytes()
        after = path.stat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DriverError(f"{label} cannot be read: {path}: {error}") from error
    if not stat.S_ISREG(after.st_mode) or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise DriverError(f"{label} changed while being read: {path}")
    if len(payload) != before.st_size:
        raise DriverError(f"{label} was truncated while being read: {path}")
    return resolved, payload, _sha256(payload)


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DriverError(f"{label} must be an object")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DriverError(f"{label} must be a nonempty string")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise DriverError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise DriverError(f"{label} must be <= {maximum}")
    return value


def _require_finite(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DriverError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        comparator = "finite" if minimum is None else f"finite and >= {minimum:g}"
        raise DriverError(f"{label} must be {comparator}")
    return parsed


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DriverError(f"JSON object has duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise DriverError(f"JSON has non-finite constant {value!r}")


def load_workload(path: Path) -> Workload:
    """Load one bounded immutable exact-output workload document."""
    resolved, payload, sha256 = _sha256_file(path, maximum_bytes=MAX_WORKLOAD_BYTES, label="workload")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise DriverError(f"workload is not UTF-8: {resolved}") from error
    except json.JSONDecodeError as error:
        raise DriverError(f"workload is malformed JSON: {error}") from error
    values = _require_mapping(decoded, "workload")
    required = {
        "schema_version",
        "case",
        "model_id",
        "model_revision",
        "prompt",
        "prompt_token_ids",
        "output_token_ids",
        "output_text",
        "finish_reason",
        "sampling",
    }
    if set(values) != required:
        missing = sorted(required - set(values))
        extra = sorted(set(values) - required)
        fragments = ([] if not missing else ["missing " + ", ".join(missing)]) + (
            [] if not extra else ["unsupported " + ", ".join(extra)]
        )
        raise DriverError("workload fields differ: " + "; ".join(fragments))
    if values["schema_version"] != WORKLOAD_SCHEMA_VERSION:
        raise DriverError(f"workload.schema_version must be {WORKLOAD_SCHEMA_VERSION!r}")
    if values["model_id"] != MODEL_ID or values["model_revision"] != MODEL_REVISION:
        raise DriverError("workload must pin the N06-A Qwen2.5-3B model identity")
    case = _require_string(values["case"], "workload.case")
    if not CASE_RE.fullmatch(case):
        raise DriverError("workload.case must be a compact marker-safe identifier")
    prompt = _require_string(values["prompt"], "workload.prompt")
    output_text = values["output_text"]
    if not isinstance(output_text, str):
        raise DriverError("workload.output_text must be a string")
    finish_reason = _require_string(values["finish_reason"], "workload.finish_reason")
    if finish_reason != "length":
        raise DriverError("N06-A fixed workload finish_reason must be 'length'")

    def token_ids(raw: object, label: str, *, minimum: int, maximum: int) -> tuple[int, ...]:
        if not isinstance(raw, list) or not minimum <= len(raw) <= maximum:
            raise DriverError(f"{label} must have {minimum}..{maximum} raw IDs")
        return tuple(_require_int(item, f"{label}[{index}]", maximum=2**32 - 1) for index, item in enumerate(raw))

    prompt_ids = token_ids(values["prompt_token_ids"], "workload.prompt_token_ids", minimum=1, maximum=MAX_PROMPT_TOKENS)
    output_ids = token_ids(values["output_token_ids"], "workload.output_token_ids", minimum=2, maximum=MAX_OUTPUT_TOKENS)
    sampling = _require_mapping(values["sampling"], "workload.sampling")
    if set(sampling) != {"temperature", "top_p"}:
        raise DriverError("workload.sampling must contain exactly temperature and top_p")
    temperature = _require_finite(sampling["temperature"], "workload.sampling.temperature", minimum=0.0)
    top_p = _require_finite(sampling["top_p"], "workload.sampling.top_p", minimum=0.0)
    if temperature != 0.0 or top_p != 1.0:
        raise DriverError("N06-A fixed workload requires greedy temperature=0.0 and top_p=1.0")
    return Workload(
        source_path=resolved,
        source_sha256=sha256,
        case=case,
        prompt=prompt,
        prompt_token_ids=prompt_ids,
        output_token_ids=output_ids,
        output_text=output_text,
        finish_reason=finish_reason,
        temperature=temperature,
        top_p=top_p,
    )


def attempt_context_from_environment(environ: Mapping[str, str] | None = None) -> AttemptContext:
    """Read the N01-only child environment without accepting an implicit order."""
    source = os.environ if environ is None else environ
    phase = source.get(N01_PHASE_ENV)
    raw_index = source.get(N01_INDEX_ENV)
    if phase not in PHASES:
        raise DriverError(
            f"{N01_PHASE_ENV} must be one of {sorted(PHASES)!r}; run through n01_repeat_control.py"
        )
    if raw_index is None or not re.fullmatch(r"[1-9][0-9]*", raw_index):
        raise DriverError(
            f"{N01_INDEX_ENV} must be a one-based decimal attempt index; run through n01_repeat_control.py"
        )
    index = int(raw_index, 10)
    if index > MAX_REQUESTS:
        raise DriverError(f"{N01_INDEX_ENV} exceeds bounded attempt index {MAX_REQUESTS}")
    return AttemptContext(phase=phase, index=index)


def _resolve_existing_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise DriverError(f"{label} cannot be resolved: {path}") from error
    if not resolved.is_dir():
        raise DriverError(f"{label} must be a directory: {path}")
    return resolved


def _model_identity_relative_path(value: object, label: str) -> str:
    """Accept one manifest path only when it stays below the model root."""
    raw = _require_string(value, label)
    if "\\" in raw:
        raise DriverError(f"{label} must use a portable relative POSIX path")
    parsed = PurePosixPath(raw)
    if (
        parsed.is_absolute()
        or raw in {".", ".."}
        or not parsed.parts
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise DriverError(f"{label} must be a non-traversing relative path")
    return raw


def _model_identity_file_path(model_path: Path, relative_path: str, *, label: str) -> Path:
    candidate = model_path.joinpath(*PurePosixPath(relative_path).parts)
    try:
        before = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise DriverError(f"{label} cannot be inspected: {candidate}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise DriverError(f"{label} must be a regular non-symlink file: {candidate}")
    try:
        resolved.relative_to(model_path)
    except ValueError as error:
        raise DriverError(f"{label} resolves outside its pinned model path") from error
    return resolved


def _load_model_identity_manifest(path: Path, *, model_path: Path) -> ModelIdentityManifest:
    """Load a bounded model identity manifest without hashing checkpoint shards.

    The manifest carries the upstream LFS object identity for large shards and
    hashes every small metadata file.  This gives a repeat an immutable model
    receipt while avoiding a 6+ GiB reread before every paired lane.
    """
    source_path, payload, source_sha256 = _sha256_file(
        path,
        maximum_bytes=MAX_MODEL_IDENTITY_MANIFEST_BYTES,
        label="model-identity-manifest",
    )
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise DriverError("model-identity-manifest is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise DriverError(f"model-identity-manifest is malformed JSON: {error}") from error
    manifest = _require_mapping(decoded, "model-identity-manifest")
    required = {
        "schema_version",
        "model_id",
        "model_revision",
        "model_path",
        "metadata_files",
        "shards",
    }
    if set(manifest) != required:
        raise DriverError("model-identity-manifest fields differ from the N06-A schema")
    if manifest["schema_version"] != MODEL_IDENTITY_MANIFEST_SCHEMA_VERSION:
        raise DriverError(
            "model-identity-manifest.schema_version must be "
            f"{MODEL_IDENTITY_MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest["model_id"] != MODEL_ID or manifest["model_revision"] != MODEL_REVISION:
        raise DriverError("model-identity-manifest must pin the N06-A Qwen2.5-3B revision")
    if _require_string(manifest["model_path"], "model-identity-manifest.model_path") != str(model_path):
        raise DriverError("model-identity-manifest.model_path must exactly equal --model-path")

    raw_metadata = manifest["metadata_files"]
    raw_shards = manifest["shards"]
    if (
        not isinstance(raw_metadata, list)
        or not raw_metadata
        or len(raw_metadata) > 128
        or not isinstance(raw_shards, list)
        or not raw_shards
        or len(raw_shards) > 128
    ):
        raise DriverError("model-identity-manifest must contain 1..128 metadata_files and 1..128 shards")
    paths: set[str] = set()
    metadata_files: list[ModelIdentityMetadataFile] = []
    for index, item in enumerate(raw_metadata):
        value = _require_mapping(item, f"model-identity-manifest.metadata_files[{index}]")
        if set(value) != {"path", "size_bytes", "sha256"}:
            raise DriverError(f"model-identity-manifest.metadata_files[{index}] fields differ")
        relative_path = _model_identity_relative_path(value["path"], f"model metadata path[{index}]")
        if relative_path in paths:
            raise DriverError("model-identity-manifest repeats a model file path")
        paths.add(relative_path)
        size_bytes = _require_int(
            value["size_bytes"], f"model metadata size_bytes[{index}]", minimum=1, maximum=MAX_MODEL_IDENTITY_METADATA_BYTES
        )
        sha256 = _require_string(value["sha256"], f"model metadata sha256[{index}]")
        if not SHA256_RE.fullmatch(sha256):
            raise DriverError(f"model metadata sha256[{index}] must be lowercase SHA-256")
        metadata_files.append(ModelIdentityMetadataFile(relative_path, size_bytes, sha256))
    shards: list[ModelIdentityShard] = []
    for index, item in enumerate(raw_shards):
        value = _require_mapping(item, f"model-identity-manifest.shards[{index}]")
        if set(value) != {"path", "size_bytes", "lfs_oid_sha256"}:
            raise DriverError(f"model-identity-manifest.shards[{index}] fields differ")
        relative_path = _model_identity_relative_path(value["path"], f"model shard path[{index}]")
        if relative_path in paths:
            raise DriverError("model-identity-manifest repeats a model file path")
        paths.add(relative_path)
        size_bytes = _require_int(value["size_bytes"], f"model shard size_bytes[{index}]", minimum=1)
        lfs_oid_sha256 = _require_string(value["lfs_oid_sha256"], f"model shard LFS OID[{index}]")
        if not SHA256_RE.fullmatch(lfs_oid_sha256):
            raise DriverError(f"model shard LFS OID[{index}] must be lowercase SHA-256")
        shards.append(ModelIdentityShard(relative_path, size_bytes, lfs_oid_sha256))
    result = ModelIdentityManifest(
        source_path=source_path,
        source_sha256=source_sha256,
        model_path=model_path,
        metadata_files=tuple(metadata_files),
        shards=tuple(shards),
    )
    _validate_model_identity_files(result, label="model-identity-manifest")
    return result


def _validate_model_identity_files(manifest: ModelIdentityManifest, *, label: str) -> None:
    """Validate small-file hashes and shard sizes against the current checkout."""
    for entry in manifest.metadata_files:
        path = _model_identity_file_path(manifest.model_path, entry.relative_path, label=f"{label} metadata {entry.relative_path}")
        resolved, payload, observed_sha256 = _sha256_file(
            path,
            maximum_bytes=MAX_MODEL_IDENTITY_METADATA_BYTES,
            label=f"{label} metadata {entry.relative_path}",
        )
        if len(payload) != entry.size_bytes or observed_sha256 != entry.sha256:
            raise DriverError(f"{label} metadata identity differs for {entry.relative_path}")
        if resolved != path:
            raise DriverError(f"{label} metadata path changed while being validated: {entry.relative_path}")
    for entry in manifest.shards:
        path = _model_identity_file_path(manifest.model_path, entry.relative_path, label=f"{label} shard {entry.relative_path}")
        try:
            before = path.stat()
            after = path.stat()
        except OSError as error:
            raise DriverError(f"{label} shard cannot be statted: {entry.relative_path}: {error}") from error
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size != entry.size_bytes
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise DriverError(f"{label} shard identity differs for {entry.relative_path}")


def _model_identity_git_command(
    *, git: Path, model_path: Path, arguments: Sequence[str], monotonic_ns: Callable[[], int]
) -> dict[str, Any]:
    argv = [str(git), "-C", str(model_path), *arguments]
    started_ns = monotonic_ns()
    receipt: dict[str, Any] = {
        "argv": argv,
        "started_ns": started_ns,
        "timeout_seconds": MODEL_IDENTITY_GIT_TIMEOUT_SECONDS,
    }
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            timeout=MODEL_IDENTITY_GIT_TIMEOUT_SECONDS,
            shell=False,
            check=False,
        )
        if (
            len(completed.stdout) > MAX_MODEL_IDENTITY_GIT_OUTPUT_BYTES
            or len(completed.stderr) > MAX_MODEL_IDENTITY_GIT_OUTPUT_BYTES
        ):
            raise DriverError("model identity git/LFS command output exceeds 1MiB")
        receipt.update(
            returncode=completed.returncode,
            stdout_text=completed.stdout.decode("utf-8"),
            stderr_text=completed.stderr.decode("utf-8"),
        )
    except BaseException as error:
        receipt.update(returncode=None, stdout_text="", stderr_text="", error=f"{type(error).__name__}: {error}")
    receipt["finished_ns"] = monotonic_ns()
    for stream in ("stdout", "stderr"):
        text = receipt[f"{stream}_text"]
        assert isinstance(text, str)
        payload = text.encode("utf-8")
        receipt[f"{stream}_bytes"] = len(payload)
        receipt[f"{stream}_sha256"] = _sha256(payload)
    return receipt


def _parse_git_lfs_oid_listing(text: str, *, label: str) -> dict[str, str]:
    """Parse `git lfs ls-files -l` without accepting ambiguous duplicate paths."""
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})\s+[ *-]\s+(.+)", raw_line)
        if match is None:
            raise DriverError(f"{label} line {line_number} is not a git-lfs long listing")
        oid, raw_path = match.groups()
        relative_path = _model_identity_relative_path(raw_path, f"{label} path on line {line_number}")
        if relative_path in result:
            raise DriverError(f"{label} repeats path {relative_path!r}")
        result[relative_path] = oid
    return result


def _default_model_identity_validator(
    *, manifest: ModelIdentityManifest, git: Path, git_sha256: str, monotonic_ns: Callable[[], int]
) -> Mapping[str, Any]:
    """Bind LFS metadata to the large shard OIDs without rereading weights."""
    head = _model_identity_git_command(
        git=git,
        model_path=manifest.model_path,
        arguments=("rev-parse", "HEAD"),
        monotonic_ns=monotonic_ns,
    )
    lfs = _model_identity_git_command(
        git=git,
        model_path=manifest.model_path,
        arguments=("lfs", "ls-files", "-l"),
        monotonic_ns=monotonic_ns,
    )
    receipt: dict[str, Any] = {
        "schema_version": MODEL_IDENTITY_VALIDATION_SCHEMA_VERSION,
        "manifest_sha256": manifest.source_sha256,
        "model_path": str(manifest.model_path),
        "model_revision": MODEL_REVISION,
        "git": {"path": str(git), "sha256": git_sha256},
        "commands": [head, lfs],
        "observed_git_head": None,
        "observed_lfs_oids": {},
        "validated": False,
        "errors": [],
    }
    if head.get("error") is not None or lfs.get("error") is not None:
        receipt["errors"].append("model identity git/LFS command failed")
        return receipt
    if head.get("returncode") != 0 or lfs.get("returncode") != 0:
        receipt["errors"].append("model identity git/LFS command returned non-zero")
        return receipt
    head_text = head["stdout_text"]
    lfs_text = lfs["stdout_text"]
    assert isinstance(head_text, str) and isinstance(lfs_text, str)
    head_lines = [line for line in head_text.splitlines() if line]
    if len(head_lines) != 1 or head_lines[0] != MODEL_REVISION:
        receipt["errors"].append("model identity git HEAD differs from pinned model revision")
        return receipt
    receipt["observed_git_head"] = head_lines[0]
    try:
        observed = _parse_git_lfs_oid_listing(lfs_text, label="model identity git-lfs listing")
    except DriverError as error:
        receipt["errors"].append(str(error))
        return receipt
    expected = {entry.relative_path: entry.lfs_oid_sha256 for entry in manifest.shards}
    missing = sorted(path for path in expected if observed.get(path) != expected[path])
    if missing:
        receipt["errors"].append("model identity git-lfs OID differs for " + ", ".join(missing))
        return receipt
    receipt["observed_lfs_oids"] = {path: observed[path] for path in sorted(expected)}
    receipt["validated"] = True
    return receipt


def _validate_model_identity_validation_receipt(
    receipt: Mapping[str, Any], *, manifest: ModelIdentityManifest, git: Path, git_sha256: str
) -> None:
    """Check the complete, no-shell Git/LFS evidence before it is promoted."""
    expected_top = {
        "schema_version",
        "manifest_sha256",
        "model_path",
        "model_revision",
        "git",
        "commands",
        "observed_git_head",
        "observed_lfs_oids",
        "validated",
        "errors",
    }
    if set(receipt) != expected_top:
        raise DriverError("model identity validation receipt fields differ")
    if (
        receipt.get("schema_version") != MODEL_IDENTITY_VALIDATION_SCHEMA_VERSION
        or receipt.get("manifest_sha256") != manifest.source_sha256
        or receipt.get("model_path") != str(manifest.model_path)
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("validated") is not True
        or receipt.get("errors") != []
    ):
        raise DriverError("model identity validation receipt does not prove the pinned model")
    if receipt.get("git") != {"path": str(git), "sha256": git_sha256}:
        raise DriverError("model identity validation receipt git executable differs")
    commands = receipt.get("commands")
    if not isinstance(commands, list) or len(commands) != 2:
        raise DriverError("model identity validation receipt lacks exact git/LFS commands")
    expected_argvs = (
        [str(git), "-C", str(manifest.model_path), "rev-parse", "HEAD"],
        [str(git), "-C", str(manifest.model_path), "lfs", "ls-files", "-l"],
    )
    for index, (command, expected_argv) in enumerate(zip(commands, expected_argvs, strict=True)):
        if not isinstance(command, Mapping) or command.get("argv") != expected_argv:
            raise DriverError(f"model identity validation command[{index}] argv differs")
        if command.get("returncode") != 0 or "error" in command:
            raise DriverError(f"model identity validation command[{index}] did not succeed")
        timeout = command.get("timeout_seconds")
        if timeout != MODEL_IDENTITY_GIT_TIMEOUT_SECONDS:
            raise DriverError(f"model identity validation command[{index}] timeout differs")
        started_ns = command.get("started_ns")
        finished_ns = command.get("finished_ns")
        if (
            isinstance(started_ns, bool)
            or not isinstance(started_ns, int)
            or isinstance(finished_ns, bool)
            or not isinstance(finished_ns, int)
            or finished_ns < started_ns
        ):
            raise DriverError(f"model identity validation command[{index}] timing is invalid")
        for stream in ("stdout", "stderr"):
            text = command.get(f"{stream}_text")
            if not isinstance(text, str):
                raise DriverError(f"model identity validation command[{index}] {stream} is invalid")
            payload = text.encode("utf-8")
            if command.get(f"{stream}_bytes") != len(payload) or command.get(f"{stream}_sha256") != _sha256(payload):
                raise DriverError(f"model identity validation command[{index}] {stream} receipt differs")
    if receipt.get("observed_git_head") != MODEL_REVISION:
        raise DriverError("model identity validation receipt git HEAD differs")
    observed = receipt.get("observed_lfs_oids")
    expected_oids = {entry.relative_path: entry.lfs_oid_sha256 for entry in manifest.shards}
    if observed != {path: expected_oids[path] for path in sorted(expected_oids)}:
        raise DriverError("model identity validation receipt LFS OIDs differ")


def _resolve_existing_file(path: Path, label: str, *, executable: bool = False) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as error:
        raise DriverError(f"{label} cannot be resolved: {path}") from error
    if not stat.S_ISREG(mode) or (executable and not os.access(resolved, os.X_OK)):
        kind = "an executable regular file" if executable else "a regular file"
        raise DriverError(f"{label} must be {kind}: {path}")
    return resolved


def _positive_port(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise DriverError(f"{label} must be in 1..65535")
    return value


def _bounded_positive(value: object, label: str, *, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise DriverError(f"{label} must be in 1..{maximum}")
    return value


def _bounded_timeout(value: object, label: str) -> float:
    parsed = _require_finite(value, label, minimum=0.001)
    if parsed > MAX_TIMEOUT_SECONDS:
        raise DriverError(f"{label} must not exceed {MAX_TIMEOUT_SECONDS:g} seconds")
    return parsed


def _bounded_request_timeout(value: object, label: str) -> float:
    parsed = _bounded_timeout(value, label)
    if parsed > MAX_TOKEN_CLIENT_REQUEST_TIMEOUT_SECONDS:
        raise DriverError(
            f"{label} must not exceed the strict token-client bound "
            f"{MAX_TOKEN_CLIENT_REQUEST_TIMEOUT_SECONDS:g} seconds"
        )
    return parsed


def _bounded_gpu_sampling_interval(value: object, label: str) -> float:
    parsed = _require_finite(value, label, minimum=MIN_GPU_MEMORY_SAMPLING_INTERVAL_SECONDS)
    if parsed > MAX_GPU_MEMORY_SAMPLING_INTERVAL_SECONDS:
        raise DriverError(
            f"{label} must be in {MIN_GPU_MEMORY_SAMPLING_INTERVAL_SECONDS:g}.."
            f"{MAX_GPU_MEMORY_SAMPLING_INTERVAL_SECONDS:g} seconds"
        )
    return parsed


def _load_argv_template(path: Path, *, label: str) -> tuple[str, ...]:
    resolved, payload, _ = _sha256_file(path, maximum_bytes=MAX_WORKLOAD_BYTES, label=label)
    try:
        decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=_json_object_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DriverError(f"{label} must be UTF-8 JSON array: {resolved}: {error}") from error
    if not isinstance(decoded, list) or not decoded or any(not isinstance(item, str) or not item for item in decoded):
        raise DriverError(f"{label} must be a nonempty string argv array")
    argv = tuple(decoded)
    if not Path(argv[0]).is_absolute():
        raise DriverError(f"{label}[0] must be an absolute executable path")
    _resolve_existing_file(Path(argv[0]), f"{label}[0]", executable=True)
    for token in argv:
        matches = set(re.findall(r"\{[^{}]+\}", token))
        if not matches <= ALLOWED_TEMPLATE_TOKENS:
            raise DriverError(f"{label} contains unsupported template token(s): {sorted(matches - ALLOWED_TEMPLATE_TOKENS)}")
        if "{" in token.replace("{port}", "").replace("{model_path}", "").replace("{model_id}", "").replace("{concurrency}", "").replace("{batch_token_budget}", "").replace("{max_model_len}", "").replace("{max_output_tokens}", "").replace("{kv_blocks}", ""):
            raise DriverError(f"{label} contains malformed template braces")
    return argv


def _expand_argv(template: Sequence[str], values: Mapping[str, str], *, label: str) -> tuple[str, ...]:
    result: list[str] = []
    for token in template:
        expanded = token
        for key, value in values.items():
            expanded = expanded.replace("{" + key + "}", value)
        if "{" in expanded or "}" in expanded:
            raise DriverError(f"{label} has an unresolved template token")
        result.append(expanded)
    return tuple(result)


def _option_values(template: Sequence[str], flag: str, *, label: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(template):
        if token == flag:
            if index + 1 >= len(template):
                raise DriverError(f"{label} has {flag} without a value")
            values.append(template[index + 1])
        elif token.startswith(flag + "="):
            values.append(token.split("=", 1)[1])
    return values



_DOCKER_RUN_VALUE_OPTIONS = frozenset({"--gpus", "--ipc", "--network", "-v", "--volume", "--name"})
_DOCKER_RUN_FLAG_OPTIONS = frozenset({"--rm"})


def _docker_run_image_operand_index(argv: Sequence[str], *, label: str) -> int:
    """Return Docker's image operand, rejecting unreviewed pre-image options.

    The exact image string must be the first non-option after ``docker run``.
    Looking merely for a digest token is insufficient: an earlier tag could be
    the image while the pinned digest becomes an inert container argument.
    """
    if len(argv) < 3 or argv[1] != "run":
        raise DriverError(f"{label} must begin with Docker run")
    index = 2
    while index < len(argv):
        token = argv[index]
        if token in _DOCKER_RUN_FLAG_OPTIONS:
            index += 1
            continue
        if token in _DOCKER_RUN_VALUE_OPTIONS:
            if index + 1 >= len(argv) or not argv[index + 1]:
                raise DriverError(f"{label} Docker option {token!r} lacks a value before its image operand")
            index += 2
            continue
        if any(
            token.startswith(option + "=")
            for option in _DOCKER_RUN_VALUE_OPTIONS
            if option.startswith("--")
        ):
            if token.endswith("="):
                raise DriverError(f"{label} Docker option {token!r} has an empty value")
            index += 1
            continue
        if token.startswith("-"):
            raise DriverError(f"{label} has unsupported Docker option before its image operand: {token!r}")
        return index
    raise DriverError(f"{label} lacks a Docker image operand")


def _require_docker_controls_before_image(
    argv: Sequence[str], *, image_operand_index: int, label: str
) -> None:
    """Reject Docker control flags that would instead reach the image entrypoint.

    Docker stops parsing its own options at the image operand.  A global scan
    for ``--gpus`` or ``-v`` therefore cannot prove the launched container had
    the requested GPU, network, IPC, cleanup, or model-mount contract.  Keep
    the accepted Docker prefix deliberately small and require every spelling
    of its control options to precede the resolved image operand.
    """
    if image_operand_index < 2 or image_operand_index >= len(argv):
        raise DriverError(f"{label} has an invalid Docker image operand index")
    for index, token in enumerate(argv[2:], start=2):
        is_value_option = token in _DOCKER_RUN_VALUE_OPTIONS
        is_equals_value_option = any(
            token.startswith(option + "=")
            for option in _DOCKER_RUN_VALUE_OPTIONS
            if option.startswith("--")
        )
        is_flag_option = token in _DOCKER_RUN_FLAG_OPTIONS or token.startswith("--rm=")
        if not (is_value_option or is_equals_value_option or is_flag_option):
            continue
        if index >= image_operand_index:
            raise DriverError(
                f"{label} Docker control option {token!r} must occur before its image operand"
            )
        if is_value_option and index + 1 >= image_operand_index:
            raise DriverError(
                f"{label} Docker control option {token!r} must retain its value before the image operand"
            )

def _require_vllm_fairness_template(template: Sequence[str], *, image_digest: str) -> None:
    """Validate the fixed vLLM fairness envelope before a server is launched."""
    if len(template) < 2 or template[1] != "run":
        raise DriverError(
            "vllm-command-json must start with an owned Docker launcher followed by the run subcommand"
        )
    if template.count("--rm") != 1:
        raise DriverError("vllm-command-json must contain exactly one Docker --rm for owned-lane cleanup")
    if any(token in {"-d", "--detach"} or token.startswith("--detach=") for token in template):
        raise DriverError("vllm-command-json must not detach the owned Docker container")
    if any(token == "--name" or token.startswith("--name=") for token in template):
        raise DriverError(
            "vllm-command-json must not set --name; the paired driver injects a unique owned container name"
        )
    if template.count(VLLM_PREFIX_CACHING_DISABLE_FLAG) != 1:
        raise DriverError(
            "vllm-command-json must contain exactly one "
            f"{VLLM_PREFIX_CACHING_DISABLE_FLAG} for the fixed-prompt comparison"
        )
    if template.count(VLLM_CHUNKED_PREFILL_ENABLE_FLAG) != 1:
        raise DriverError(
            "vllm-command-json must contain exactly one "
            f"{VLLM_CHUNKED_PREFILL_ENABLE_FLAG} for P2048/M32"
        )
    if any(token == "--attention-backend" or token.startswith("--attention-backend=") for token in template):
        raise DriverError("vllm-command-json must leave vLLM attention backend auto-selected")
    if _option_values(template, VLLM_DTYPE_FLAG, label="vllm-command-json") != [VLLM_DTYPE]:
        raise DriverError(f"vllm-command-json must set exactly one {VLLM_DTYPE_FLAG}={VLLM_DTYPE}")
    if _option_values(template, VLLM_KV_CACHE_DTYPE_FLAG, label="vllm-command-json") != [VLLM_KV_CACHE_DTYPE]:
        raise DriverError(
            f"vllm-command-json must set exactly one {VLLM_KV_CACHE_DTYPE_FLAG}={VLLM_KV_CACHE_DTYPE}"
        )
    if _option_values(template, "--ipc", label="vllm-command-json") != ["host"]:
        raise DriverError("vllm-command-json must set exactly one --ipc host")
    if (
        template.count("--network") != 1
        or any(token.startswith("--network=") for token in template)
        or _option_values(template, "--network", label="vllm-command-json") != ["host"]
    ):
        raise DriverError("vllm-command-json must set exactly one separate --network host argv pair")
    if _option_values(template, "--gpus", label="vllm-command-json") != ["device=0"]:
        raise DriverError(
            "vllm-command-json must bind Docker to physical GPU 0 with exactly one --gpus device=0"
        )
    utilization_values = _option_values(template, VLLM_GPU_MEMORY_UTILIZATION_FLAG, label="vllm-command-json")
    if utilization_values != [VLLM_GPU_MEMORY_UTILIZATION]:
        raise DriverError(
            "vllm-command-json must set exactly one "
            f"{VLLM_GPU_MEMORY_UTILIZATION_FLAG}={VLLM_GPU_MEMORY_UTILIZATION}"
        )
    expected_image = f"vllm/vllm-openai@{image_digest}"
    image_tokens = [token for token in template if image_digest in token]
    if image_tokens != [expected_image]:
        raise DriverError(
            "vllm-command-json must use exactly the image argv "
            f"{expected_image!r}; the digest cannot appear in any other argument"
        )
    image_operand_index = _docker_run_image_operand_index(template, label="vllm-command-json")
    _require_docker_controls_before_image(
        template,
        image_operand_index=image_operand_index,
        label="vllm-command-json",
    )
    if template[image_operand_index] != expected_image:
        raise DriverError(
            "vllm-command-json must place its exact pinned image as Docker's first "
            "non-option image operand"
        )
    mount_values = [
        *_option_values(template, "-v", label="vllm-command-json"),
        *_option_values(template, "--volume", label="vllm-command-json"),
    ]
    if mount_values != ["{model_path}:/model:ro"]:
        raise DriverError(
            "vllm-command-json must provide exactly one read-only model mount "
            "{model_path}:/model:ro"
        )
    if _option_values(template, "--model", label="vllm-command-json") != ["/model"]:
        raise DriverError("vllm-command-json must pass exactly one --model /model to the image entrypoint")
    if _option_values(template, "--served-model-name", label="vllm-command-json") != ["{model_id}"]:
        raise DriverError(
            "vllm-command-json must pass exactly one --served-model-name {model_id} to the image entrypoint"
        )
    expected_scheduler_options = {
        "--max-model-len": ["{max_model_len}"],
        "--max-num-seqs": ["{concurrency}"],
        "--max-num-batched-tokens": ["{batch_token_budget}"],
    }
    for flag, expected in expected_scheduler_options.items():
        if _option_values(template, flag, label="vllm-command-json") != expected:
            raise DriverError(
                f"vllm-command-json must bind exactly one {flag} to {expected[0]}"
            )


def build_config(args: argparse.Namespace) -> DriverConfig:
    workload = load_workload(args.workload)
    artifact_root = args.artifact_root.expanduser().resolve()
    if artifact_root.exists() and not artifact_root.is_dir():
        raise DriverError(f"artifact-root must be a directory or a new directory: {artifact_root}")
    model_path = _resolve_existing_directory(args.model_path, "model-path")
    model_identity = _load_model_identity_manifest(
        args.model_identity_manifest,
        model_path=model_path,
    )
    model_git = _resolve_existing_file(args.model_git, "model-git", executable=True)
    _, _, model_git_sha256 = _sha256_file(
        model_git,
        maximum_bytes=MAX_EXECUTABLE_BYTES,
        label="model-git",
    )
    riley_binary = _resolve_existing_file(args.riley_binary, "riley-binary", executable=True)
    vllm_template = _load_argv_template(args.vllm_command_json, label="vllm-command-json")
    vllm_command_path, _, vllm_command_sha256 = _sha256_file(
        args.vllm_command_json,
        maximum_bytes=MAX_WORKLOAD_BYTES,
        label="vllm-command-json",
    )
    vllm_image_digest = _require_string(args.vllm_image_digest, "vllm-image-digest")
    if not VLLM_IMAGE_DIGEST_RE.fullmatch(vllm_image_digest):
        raise DriverError("vllm-image-digest must be an immutable lowercase sha256:<64-hex> Docker digest")
    for placeholder in ("{port}", "{model_path}", "{model_id}"):
        if sum(token.count(placeholder) for token in vllm_template) != 1:
            raise DriverError(f"vllm-command-json must contain {placeholder} exactly once")
    _require_vllm_fairness_template(vllm_template, image_digest=vllm_image_digest)
    _, _, riley_binary_sha256 = _sha256_file(
        riley_binary,
        maximum_bytes=MAX_EXECUTABLE_BYTES,
        label="riley-binary",
    )
    vllm_launcher = _resolve_existing_file(Path(vllm_template[0]), "vllm command launcher", executable=True)
    _, _, vllm_launcher_sha256 = _sha256_file(
        vllm_launcher,
        maximum_bytes=MAX_EXECUTABLE_BYTES,
        label="vllm command launcher",
    )
    vllm_backend_requested = _require_string(args.vllm_backend_requested, "vllm-backend-requested")
    if any(character.isspace() for character in vllm_backend_requested):
        raise DriverError("vllm-backend-requested must be marker-safe")
    if vllm_backend_requested != VLLM_AUTO_BACKEND_REQUESTED:
        raise DriverError(
            f"vllm-backend-requested must be {VLLM_AUTO_BACKEND_REQUESTED!r}; do not force an attention backend"
        )
    try:
        vllm_backend_receipt_regex = re.compile(args.vllm_backend_receipt_regex)
    except re.error as error:
        raise DriverError(f"vllm-backend-receipt-regex is invalid: {error}") from error
    if set(vllm_backend_receipt_regex.groupindex) != {"backend_resolved"}:
        raise DriverError("vllm-backend-receipt-regex must contain exactly one named group: backend_resolved")
    if vllm_backend_receipt_regex.pattern != VLLM_BACKEND_RECEIPT_REGEX_PATTERN:
        raise DriverError(
            "vllm-backend-receipt-regex must use the source-verified vLLM 0.29 auto-selector receipt shape"
        )
    vllm_startup_required_fragments = tuple(args.vllm_startup_required_fragment)
    if not vllm_startup_required_fragments or any(
        not isinstance(fragment, str) or not fragment or "\x00" in fragment or len(fragment) > 1_024
        for fragment in vllm_startup_required_fragments
    ):
        raise DriverError("at least one bounded nonempty --vllm-startup-required-fragment is required")
    if len(set(vllm_startup_required_fragments)) != len(vllm_startup_required_fragments):
        raise DriverError("vllm startup required fragments must be unique")
    version_template = (
        _load_argv_template(args.vllm_version_command_json, label="vllm-version-command-json")
        if args.vllm_version_command_json is not None
        else None
    )
    riley_port = _positive_port(args.riley_port, "riley-port")
    vllm_port = _positive_port(args.vllm_port, "vllm-port")
    if riley_port == vllm_port:
        raise DriverError("riley-port and vllm-port must differ")
    concurrency = _bounded_positive(args.concurrency, "concurrency", maximum=MAX_CONCURRENCY)
    batch_token_budget = _bounded_positive(args.batch_token_budget, "batch-token-budget", maximum=MAX_CONCURRENCY)
    if concurrency > batch_token_budget:
        raise DriverError("concurrency must not exceed the native D128 batch-token-budget")
    retained_requests = _bounded_positive(args.retained_requests, "retained-requests", maximum=MAX_REQUESTS)
    warmup_requests = _bounded_positive(args.warmup_requests, "warmup-requests", maximum=MAX_REQUESTS)
    if retained_requests < concurrency or warmup_requests < concurrency:
        raise DriverError("retained-requests and warmup-requests must each be at least concurrency")
    max_model_len = _bounded_positive(args.max_model_len, "max-model-len", maximum=MAX_PROMPT_TOKENS)
    if max_model_len < workload.prompt_tokens + workload.max_output_tokens:
        raise DriverError("max-model-len must fit the exact prompt plus generated token count")
    kv_blocks = _bounded_positive(args.kv_blocks, "kv-blocks", maximum=1_000_000)
    required_blocks = concurrency * math.ceil((workload.prompt_tokens + workload.max_output_tokens) / 16)
    if kv_blocks < required_blocks:
        raise DriverError(
            f"kv-blocks must be at least {required_blocks} for C={concurrency} and the fixed workload"
        )
    cuda_visible_devices = _require_string(args.cuda_visible_devices, "cuda-visible-devices")
    if cuda_visible_devices != "0":
        raise DriverError(
            "cuda-visible-devices must be exactly '0' so the Riley lane and host GPU-0 sampler share one physical device"
        )
    nvidia_smi = _resolve_existing_file(args.nvidia_smi, "nvidia-smi", executable=True)
    gpu_memory_sampling_interval_seconds = _bounded_gpu_sampling_interval(
        args.gpu_memory_sampling_interval_seconds,
        "gpu-memory-sampling-interval-seconds",
    )
    return DriverConfig(
        artifact_root=artifact_root,
        workload=workload,
        model_path=model_path,
        model_identity=model_identity,
        model_git=model_git,
        model_git_sha256=model_git_sha256,
        riley_binary=riley_binary,
        vllm_template=vllm_template,
        vllm_command_path=vllm_command_path,
        vllm_command_sha256=vllm_command_sha256,
        riley_binary_sha256=riley_binary_sha256,
        vllm_launcher_sha256=vllm_launcher_sha256,
        vllm_backend_requested=vllm_backend_requested,
        vllm_backend_receipt_regex=vllm_backend_receipt_regex,
        vllm_startup_required_fragments=vllm_startup_required_fragments,
        vllm_image_digest=vllm_image_digest,
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        whole_gpu_sampled_peak_limit_bytes=WHOLE_GPU_SAMPLED_PEAK_LIMIT_BYTES,
        nvidia_smi=nvidia_smi,
        gpu_memory_sampling_interval_seconds=gpu_memory_sampling_interval_seconds,
        riley_port=riley_port,
        vllm_port=vllm_port,
        concurrency=concurrency,
        batch_token_budget=batch_token_budget,
        retained_requests=retained_requests,
        warmup_requests=warmup_requests,
        max_model_len=max_model_len,
        kv_blocks=kv_blocks,
        startup_timeout_seconds=_bounded_timeout(args.startup_timeout_seconds, "startup-timeout-seconds"),
        request_timeout_seconds=_bounded_request_timeout(args.request_timeout_seconds, "request-timeout-seconds"),
        shutdown_timeout_seconds=_bounded_timeout(args.shutdown_timeout_seconds, "shutdown-timeout-seconds"),
        cuda_visible_devices=cuda_visible_devices,
        vllm_version_template=version_template,
    )


def _write_bytes_create_only(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise DriverError(f"refusing to overwrite evidence: {path}") from error
    except OSError as error:
        raise DriverError(f"cannot write evidence {path}: {error}") from error


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    try:
        payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DriverError(f"cannot encode evidence {path}: {error}") from error
    _write_bytes_create_only(path, payload)


def _write_jsonl_create_only(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        payload = b"".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in rows
        )
    except (TypeError, ValueError) as error:
        raise DriverError(f"cannot encode raw rows {path}: {error}") from error
    _write_bytes_create_only(path, payload)


def _retained_artifacts(attempt_dir: Path, lane: str) -> RetainedArtifacts:
    """Hash the lane rows/phase and shared attempt config after they are final.

    The caller uses this only after the retained phase has been written and the
    owned server has stopped.  The marker therefore names immutable evidence
    rather than a mutable live server log.
    """
    request_rows_path, _, request_rows_sha256 = _sha256_file(
        attempt_dir / f"{lane}.requests.jsonl",
        maximum_bytes=MAX_RETAINED_ROWS_BYTES,
        label=f"{lane} retained request rows",
    )
    phase_path, _, phase_sha256 = _sha256_file(
        attempt_dir / f"{lane}.phase.json",
        maximum_bytes=MAX_WORKLOAD_BYTES,
        label=f"{lane} retained phase accounting",
    )
    attempt_config_path, _, attempt_config_sha256 = _sha256_file(
        attempt_dir / "attempt.config.json",
        maximum_bytes=MAX_WORKLOAD_BYTES,
        label="attempt configuration",
    )
    return RetainedArtifacts(
        request_rows_path=request_rows_path,
        request_rows_sha256=request_rows_sha256,
        phase_path=phase_path,
        phase_sha256=phase_sha256,
        attempt_config_path=attempt_config_path,
        attempt_config_sha256=attempt_config_sha256,
    )


def _server_warmup_artifacts(attempt_dir: Path, lane: str) -> ServerWarmupArtifacts:
    """Bind a marker to its completed, strict per-server warmup evidence."""
    request_rows_path, _, request_rows_sha256 = _sha256_file(
        attempt_dir / f"{lane}.server-warmup.requests.jsonl",
        maximum_bytes=MAX_RETAINED_ROWS_BYTES,
        label=f"{lane} server warmup request rows",
    )
    phase_path, _, phase_sha256 = _sha256_file(
        attempt_dir / f"{lane}.server-warmup.phase.json",
        maximum_bytes=MAX_WORKLOAD_BYTES,
        label=f"{lane} server warmup phase accounting",
    )
    return ServerWarmupArtifacts(
        request_rows_path=request_rows_path,
        request_rows_sha256=request_rows_sha256,
        phase_path=phase_path,
        phase_sha256=phase_sha256,
    )


def _write_whole_gpu_memory_evidence(
    *,
    attempt_dir: Path,
    lane: str,
    sample_result: Mapping[str, Any],
    peak_limit_bytes: int,
    expected_interval_seconds: float,
    server_started_ns: int,
    server_cleanup_completed_ns: int,
) -> dict[str, Any]:
    """Freeze bounded, lifecycle-covering whole-GPU samples for one lane.

    A pre/post N01 observation cannot prove a lane's peak.  This receipt binds
    the physical-GPU samples to the owned process lifetime: an initial query
    must start before launch, one query must overlap the running process, and
    the final query must finish after cleanup.  Query duration and observed
    start-to-start cadence are also eligibility conditions.
    """
    if (
        isinstance(server_started_ns, bool)
        or not isinstance(server_started_ns, int)
        or server_started_ns < 0
        or isinstance(server_cleanup_completed_ns, bool)
        or not isinstance(server_cleanup_completed_ns, int)
        or server_cleanup_completed_ns < server_started_ns
    ):
        raise DriverError(f"{lane} whole-GPU lifecycle timestamps are invalid")
    interval_seconds = _bounded_gpu_sampling_interval(
        sample_result.get("sampling_interval_seconds"),
        f"{lane} whole-GPU sampler interval",
    )
    if not math.isclose(interval_seconds, expected_interval_seconds, rel_tol=0.0, abs_tol=1e-12):
        raise DriverError(f"{lane} whole-GPU sampler interval differs from its configured interval")
    max_start_gap_seconds = interval_seconds + GPU_MEMORY_SAMPLE_SCHEDULING_SLACK_SECONDS
    maximum_sample_duration_ns = int(MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS * 1_000_000_000)
    maximum_start_gap_ns = int(max_start_gap_seconds * 1_000_000_000)
    raw_samples = sample_result.get("samples")
    raw_errors = sample_result.get("errors")
    if not isinstance(raw_samples, list) or not all(isinstance(item, Mapping) for item in raw_samples):
        raise DriverError(f"{lane} whole-GPU sampler returned invalid samples")
    if not isinstance(raw_errors, list) or not all(isinstance(item, str) for item in raw_errors):
        raise DriverError(f"{lane} whole-GPU sampler returned invalid errors")
    samples: list[dict[str, int]] = []
    validation_errors: list[str] = []
    previous_started_ns: int | None = None
    previous_finished_ns: int | None = None
    for index, raw_sample in enumerate(raw_samples):
        started_ns = raw_sample.get("started_ns")
        finished_ns = raw_sample.get("finished_ns")
        timestamp = raw_sample.get("monotonic_ns")
        used_bytes = raw_sample.get("memory_used_bytes")
        if (
            isinstance(started_ns, bool)
            or not isinstance(started_ns, int)
            or started_ns < 0
            or isinstance(finished_ns, bool)
            or not isinstance(finished_ns, int)
            or finished_ns < started_ns
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
            or isinstance(used_bytes, bool)
            or not isinstance(used_bytes, int)
            or used_bytes < 0
        ):
            raise DriverError(f"{lane} whole-GPU sample {index} is invalid")
        if timestamp != finished_ns:
            validation_errors.append(f"sample {index} monotonic_ns does not equal its finished_ns")
        if finished_ns - started_ns > maximum_sample_duration_ns:
            validation_errors.append(
                f"sample {index} duration exceeds {MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS:g}s"
            )
        if previous_started_ns is not None:
            if started_ns < previous_started_ns or (
                previous_finished_ns is not None and started_ns < previous_finished_ns
            ):
                validation_errors.append(f"sample {index} timing order overlaps or moves backwards")
            elif started_ns - previous_started_ns > maximum_start_gap_ns:
                validation_errors.append(
                    f"sample {index} start gap exceeds {max_start_gap_seconds:g}s"
                )
        previous_started_ns = started_ns
        previous_finished_ns = finished_ns
        samples.append(
            {
                "started_ns": started_ns,
                "finished_ns": finished_ns,
                "monotonic_ns": timestamp,
                "memory_used_bytes": used_bytes,
            }
        )
    sample_path = attempt_dir / f"{lane}.whole-gpu-memory.samples.jsonl"
    _write_jsonl_create_only(sample_path, samples)
    sample_resolved, _, sample_sha256 = _sha256_file(
        sample_path,
        maximum_bytes=MAX_WORKLOAD_BYTES,
        label=f"{lane} whole-GPU memory samples",
    )
    peak_used_bytes = max((item["memory_used_bytes"] for item in samples), default=None)
    sample_started_before_server_launch = bool(samples) and samples[0]["started_ns"] <= server_started_ns
    sample_overlapped_server_lifetime = any(
        sample["started_ns"] <= server_cleanup_completed_ns
        and sample["finished_ns"] >= server_started_ns
        for sample in samples
    )
    sample_finished_after_server_cleanup = bool(samples) and samples[-1]["finished_ns"] >= server_cleanup_completed_ns
    if not sample_started_before_server_launch:
        validation_errors.append("no sample started before the owned server launch")
    if not sample_overlapped_server_lifetime:
        validation_errors.append("no sample query overlapped the owned server lifetime")
    if not sample_finished_after_server_cleanup:
        validation_errors.append("no sample finished after owned server cleanup")
    all_errors = [*raw_errors, *validation_errors]
    compliant = (
        len(samples) >= 2
        and not all_errors
        and peak_used_bytes is not None
        and peak_used_bytes <= peak_limit_bytes
    )
    lifecycle = {
        "server_started_ns": server_started_ns,
        "server_cleanup_completed_ns": server_cleanup_completed_ns,
        "sample_started_before_server_launch": sample_started_before_server_launch,
        "sample_overlapped_server_lifetime": sample_overlapped_server_lifetime,
        "sample_finished_after_server_cleanup": sample_finished_after_server_cleanup,
    }
    peak_record = {
        "schema_version": "riley.n06a-whole-gpu-memory-peak.v1",
        "lane": lane,
        "gpu_index": 0,
        "sampling_interval_seconds": interval_seconds,
        "max_sample_duration_seconds": MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS,
        "max_start_gap_seconds": max_start_gap_seconds,
        "sample_count": len(samples),
        "sample_path": str(sample_resolved),
        "sample_sha256": sample_sha256,
        "peak_used_bytes": peak_used_bytes,
        "peak_limit_bytes": peak_limit_bytes,
        "compliant": compliant,
        "errors": all_errors,
        "lifecycle": lifecycle,
    }
    peak_path = attempt_dir / f"{lane}.whole-gpu-memory.peak.json"
    _write_json_create_only(peak_path, peak_record)
    peak_resolved, _, peak_sha256 = _sha256_file(
        peak_path,
        maximum_bytes=MAX_WORKLOAD_BYTES,
        label=f"{lane} whole-GPU memory peak receipt",
    )
    return {
        "sample_path": str(sample_resolved),
        "sample_sha256": sample_sha256,
        "peak_path": str(peak_resolved),
        "peak_sha256": peak_sha256,
        "sample_count": len(samples),
        "peak_used_bytes": peak_used_bytes,
        "peak_limit_bytes": peak_limit_bytes,
        "compliant": compliant,
        "errors": all_errors,
        "sampling_interval_seconds": interval_seconds,
        "max_sample_duration_seconds": MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS,
        "max_start_gap_seconds": max_start_gap_seconds,
        "lifecycle": lifecycle,
    }


def _create_attempt_directory(config: DriverConfig, context: AttemptContext) -> Path:
    root = config.artifact_root
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DriverError(f"cannot create artifact-root {root}: {error}") from error
    if not root.is_dir():
        raise DriverError(f"artifact-root is not a directory: {root}")
    target = root / f"{context.phase}-{context.index:03d}"
    try:
        target.mkdir()
    except FileExistsError as error:
        raise DriverError(f"refusing to overwrite existing attempt evidence: {target}") from error
    except OSError as error:
        raise DriverError(f"cannot create attempt artifact directory {target}: {error}") from error
    return target.resolve(strict=True)


def _freeze_workload_snapshot(config: DriverConfig, attempt_dir: Path) -> tuple[Path, str]:
    """Copy the already-validated workload into the create-only attempt evidence.

    A later offline summary must not depend on a transient model staging or
    `/tmp` workload path remaining available.  Rehash the source immediately
    before the copy so a file replacement between configuration and launch
    cannot silently change the reference used for replay.
    """
    _, payload, observed_sha256 = _sha256_file(
        config.workload.source_path,
        maximum_bytes=MAX_WORKLOAD_BYTES,
        label="validated N06-A workload source",
    )
    if observed_sha256 != config.workload.source_sha256:
        raise DriverError("validated N06-A workload source changed before attempt snapshot")
    destination = attempt_dir / "workload.pinned.json"
    _write_bytes_create_only(destination, payload)
    resolved, _, snapshot_sha256 = _sha256_file(
        destination,
        maximum_bytes=MAX_WORKLOAD_BYTES,
        label="pinned N06-A workload snapshot",
    )
    if snapshot_sha256 != config.workload.source_sha256:
        raise DriverError("pinned N06-A workload snapshot hash differs from validated source")
    return resolved, snapshot_sha256


def _freeze_model_identity_snapshot(config: DriverConfig, attempt_dir: Path) -> tuple[Path, str]:
    """Freeze the manifest after revalidating the live model identity.

    The pinned copy makes an offline summary independent of an operator's
    source-manifest staging path.  We still validate the source hash and live
    metadata/shard sizes immediately before the lane starts.
    """
    _, payload, observed_sha256 = _sha256_file(
        config.model_identity.source_path,
        maximum_bytes=MAX_MODEL_IDENTITY_MANIFEST_BYTES,
        label="validated model-identity-manifest source",
    )
    if observed_sha256 != config.model_identity.source_sha256:
        raise DriverError("validated model-identity-manifest source changed before attempt snapshot")
    _validate_model_identity_files(config.model_identity, label="validated model-identity-manifest")
    destination = attempt_dir / "model.identity.manifest.json"
    _write_bytes_create_only(destination, payload)
    snapshot = _load_model_identity_manifest(destination, model_path=config.model_path)
    if snapshot.source_sha256 != config.model_identity.source_sha256:
        raise DriverError("pinned model-identity-manifest hash differs from validated source")
    return snapshot.source_path, snapshot.source_sha256


def _write_model_identity_validation(
    *, config: DriverConfig, attempt_dir: Path, dependencies: DriverDependencies
) -> tuple[Path, str]:
    """Retain the per-attempt Git/LFS identity receipt alongside the manifest."""
    validator = dependencies.model_identity_validator or _default_model_identity_validator
    receipt = validator(
        manifest=config.model_identity,
        git=config.model_git,
        git_sha256=config.model_git_sha256,
        monotonic_ns=dependencies.monotonic_ns,
    )
    _validate_model_identity_validation_receipt(
        receipt,
        manifest=config.model_identity,
        git=config.model_git,
        git_sha256=config.model_git_sha256,
    )
    destination = attempt_dir / "model.identity.validation.json"
    _write_json_create_only(destination, dict(receipt))
    resolved, _, sha256 = _sha256_file(
        destination,
        maximum_bytes=MAX_MODEL_IDENTITY_MANIFEST_BYTES,
        label="pinned model identity validation receipt",
    )
    return resolved, sha256


def _riley_argv(config: DriverConfig) -> tuple[str, ...]:
    workload = config.workload
    return (
        str(config.riley_binary),
        "serve",
        "--model",
        str(config.model_path),
        "--model-id",
        MODEL_ID,
        "--bind",
        f"127.0.0.1:{config.riley_port}",
        "--device",
        "0",
        "--max-active-sequences",
        str(config.concurrency),
        "--max-waiting-requests",
        str(config.concurrency),
        "--max-sequence-tokens",
        str(config.max_model_len),
        "--max-output-tokens",
        str(workload.max_output_tokens),
        "--batch-token-budget",
        str(config.batch_token_budget),
        "--prefill-chunk-tokens",
        str(config.batch_token_budget),
        "--kv-blocks",
        str(config.kv_blocks),
        "--execution-graph-policy",
        "disabled",
        "--decode-attention-backend",
        REQUESTED_BACKEND_CLI_ID,
        "--reduction-profile",
        "canonical-v1",
        "--sampling-backend",
        "gpu-greedy",
        "--max-weight-bytes",
        str(8 * 1024 * 1024 * 1024),
        "--shutdown-on-stdin",
    )


def _vllm_argv(config: DriverConfig) -> tuple[str, ...]:
    return _expand_argv(
        config.vllm_template,
        {
            "port": str(config.vllm_port),
            "model_path": str(config.model_path),
            "model_id": MODEL_ID,
            "concurrency": str(config.concurrency),
            "batch_token_budget": str(config.batch_token_budget),
            "max_model_len": str(config.max_model_len),
            "max_output_tokens": str(config.workload.max_output_tokens),
            "kv_blocks": str(config.kv_blocks),
        },
        label="vllm-command-json",
    )


def _vllm_container_name(attempt_dir: Path, context: AttemptContext) -> str:
    """Derive a reproducible, artifact-root-scoped Docker container identity.

    Docker names must be visible to the daemon before a process-group signal is
    meaningful.  The attempt directory is create-only, and its resolved path
    makes this name deterministic for one retry while avoiding a fixed global
    name that could collide with another benchmark artifact root.
    """
    digest = _sha256(str(attempt_dir.resolve(strict=True)).encode("utf-8"))[:20]
    name = f"riley-n06a-vllm-{context.phase}-{context.index:04d}-{digest}"
    if not DOCKER_CONTAINER_NAME_RE.fullmatch(name):
        raise DriverError("derived vLLM Docker container name is invalid")
    return name


def _vllm_argv_for_container(config: DriverConfig, container_name: str) -> tuple[str, ...]:
    """Inject exactly one non-detached, owned Docker name into a validated argv."""
    if not DOCKER_CONTAINER_NAME_RE.fullmatch(container_name):
        raise DriverError("vLLM Docker container name is invalid")
    argv = list(_vllm_argv(config))
    if len(argv) < 2 or argv[1] != "run":
        raise DriverError("expanded vLLM argv does not retain Docker run as its subcommand")
    if argv.count("--rm") != 1:
        raise DriverError("expanded vLLM argv does not retain exactly one Docker --rm")
    if any(token in {"-d", "--detach"} or token.startswith("--detach=") for token in argv):
        raise DriverError("expanded vLLM argv must not detach the owned Docker container")
    if any(token == "--name" or token.startswith("--name=") for token in argv):
        raise DriverError("expanded vLLM argv unexpectedly contains a Docker container name")
    return tuple([*argv[:2], "--name", container_name, *argv[2:]])


def _child_environment(config: DriverConfig) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    return environment


def _launch_server(
    *,
    lane: str,
    argv: Sequence[str],
    attempt_dir: Path,
    dependencies: DriverDependencies,
    environment: Mapping[str, str],
    use_shutdown_stdin: bool,
) -> ManagedProcess:
    stdout_path = attempt_dir / f"{lane}.stdout.log"
    stderr_path = attempt_dir / f"{lane}.stderr.log"
    stdout_stream: TextIO | Any | None = None
    try:
        stdout_stream = stdout_path.open("xb")
        stderr_stream = stderr_path.open("xb")
    except OSError as error:
        if stdout_stream is not None:
            with contextlib.suppress(Exception):
                stdout_stream.close()
        raise DriverError(f"cannot create {lane} startup logs: {error}") from error
    try:
        process = dependencies.popen(
            list(argv),
            cwd=str(attempt_dir),
            env=dict(environment),
            stdin=subprocess.PIPE if use_shutdown_stdin else subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
            shell=False,
        )
    except BaseException as error:
        stdout_stream.close()
        stderr_stream.close()
        raise DriverError(f"cannot launch {lane}: {type(error).__name__}: {error}") from error
    return ManagedProcess(
        lane=lane,
        process=process,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_stream=stdout_stream,
        stderr_stream=stderr_stream,
        started_ns=dependencies.monotonic_ns(),
        command=tuple(argv),
        used_shutdown_stdin=use_shutdown_stdin,
    )


def _probe_ready(port: int, endpoint: str) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
    try:
        connection.request("GET", endpoint, headers={"Connection": "close"})
        response = connection.getresponse()
        body = response.read(64 * 1024 + 1)
        if len(body) > 64 * 1024:
            raise DriverError("readiness response exceeds 64KiB")
        return response.status, body
    finally:
        connection.close()


def wait_ready(process: ProcessLike, port: int, lane: str, timeout_seconds: float) -> Mapping[str, Any]:
    """Wait only for the owned localhost process and retain the observed endpoint."""
    deadline = time.monotonic() + timeout_seconds
    endpoints = ("/readyz",) if lane == "riley" else ("/health", "/readyz")
    last_error: str | None = None
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise DriverError(f"{lane} exited before readiness with code {return_code}")
        for endpoint in endpoints:
            try:
                status, body = _probe_ready(port, endpoint)
                if status == 200:
                    if lane == "riley":
                        try:
                            payload = json.loads(body.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as error:
                            raise DriverError(f"Riley readyz returned malformed JSON: {error}") from error
                        if not isinstance(payload, dict) or payload.get("ready") is not True or payload.get("accepting") is not True:
                            raise DriverError("Riley readyz did not confirm ready and accepting")
                    return {"endpoint": endpoint, "http_status": status, "body_sha256": _sha256(body)}
                last_error = f"{endpoint} returned HTTP {status}"
            except DriverError:
                raise
            except Exception as error:
                last_error = f"{endpoint}: {type(error).__name__}: {error}"
        time.sleep(0.05)
    raise DriverError(f"{lane} did not become ready within {timeout_seconds:g}s; last observation: {last_error}")


def _parse_marker_fields(line: str, *, prefix: str, label: str) -> dict[str, str]:
    if not line.startswith(prefix + " "):
        raise DriverError(f"{label} must start with {prefix}")
    fields: dict[str, str] = {}
    for item in line[len(prefix) + 1 :].split():
        if "=" not in item:
            raise DriverError(f"{label} has malformed token {item!r}")
        key, value = item.split("=", 1)
        if not key or not value or key in fields:
            raise DriverError(f"{label} has malformed or duplicate token {item!r}")
        fields[key] = value
    return fields


def capture_riley_startup_snapshot(stderr_path: Path, destination: Path) -> StartupSnapshot:
    """Copy current Riley stderr once and validate its actual D128 startup receipt."""
    _, payload, sha256 = _sha256_file(stderr_path, maximum_bytes=MAX_STARTUP_LOG_BYTES, label="Riley stderr")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DriverError("Riley stderr is not UTF-8 before startup snapshot") from error
    receipts: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        offset = line.find(STARTUP_RECEIPT_PREFIX)
        if offset < 0:
            continue
        if line.find(STARTUP_RECEIPT_PREFIX, offset + len(STARTUP_RECEIPT_PREFIX)) >= 0:
            raise DriverError(f"Riley stderr line {line_number} contains two startup receipts")
        receipts.append(
            _parse_marker_fields(
                line[offset:],
                prefix=STARTUP_RECEIPT_PREFIX,
                label=f"Riley stderr line {line_number}",
            )
        )
    if len(receipts) != 1:
        raise DriverError("Riley startup stderr must contain exactly one D128 startup receipt before measurement")
    receipt = receipts[0]
    expected_fields = {
        "requested_backend",
        "resolved_ragged_backend",
        "fallback_reason",
        "query_heads",
        "key_value_heads",
        "head_size",
        "page_size",
        "graph",
    }
    if set(receipt) != expected_fields:
        raise DriverError("Riley startup receipt fields differ from N06-A contract")
    expected_values = {
        "requested_backend": REQUESTED_BACKEND_CLI_ID,
        "resolved_ragged_backend": RESOLVED_BACKEND_ID,
        "fallback_reason": "none",
        "query_heads": "16",
        "key_value_heads": "2",
        "head_size": "128",
        "page_size": "16",
        "graph": "false",
    }
    if receipt != expected_values:
        raise DriverError("Riley startup receipt did not resolve the expected graph-disabled D128 backend")
    _write_bytes_create_only(destination, payload)
    return StartupSnapshot(path=destination.resolve(strict=True), sha256=sha256, receipt=receipt)


def _capture_log_snapshot(source: Path, destination: Path, *, label: str) -> tuple[Path, bytes, str]:
    _, payload, sha256 = _sha256_file(source, maximum_bytes=MAX_STARTUP_LOG_BYTES, label=label)
    _write_bytes_create_only(destination, payload)
    return destination.resolve(strict=True), payload, sha256


def capture_vllm_startup_snapshot(
    *,
    stdout_path: Path,
    stderr_path: Path,
    stdout_destination: Path,
    stderr_destination: Path,
    backend_requested: str,
    backend_receipt_regex: re.Pattern[str],
    required_fragments: Sequence[str],
) -> VllmStartupSnapshot:
    """Freeze both vLLM streams at readiness and require one actual backend match."""
    stdout_snapshot, stdout_payload, stdout_sha256 = _capture_log_snapshot(
        stdout_path,
        stdout_destination,
        label="vLLM stdout",
    )
    stderr_snapshot, stderr_payload, stderr_sha256 = _capture_log_snapshot(
        stderr_path,
        stderr_destination,
        label="vLLM stderr",
    )
    try:
        startup_text = (stdout_payload + b"\n" + stderr_payload).decode("utf-8")
    except UnicodeDecodeError as error:
        raise DriverError("vLLM startup stdout/stderr is not UTF-8") from error
    missing_fragments = [fragment for fragment in required_fragments if fragment not in startup_text]
    if missing_fragments:
        raise DriverError("vLLM startup logs lack required attestation fragment(s): " + ", ".join(repr(item) for item in missing_fragments))
    matches = list(backend_receipt_regex.finditer(startup_text))
    if len(matches) != 1:
        raise DriverError(
            "vLLM startup logs must contain exactly one backend receipt regex match; "
            f"observed {len(matches)}"
        )
    backend_resolved = matches[0].group("backend_resolved")
    if not backend_resolved or any(character.isspace() for character in backend_resolved):
        raise DriverError("vLLM backend_resolved capture must be a nonempty marker-safe value")
    return VllmStartupSnapshot(
        stdout_path=stdout_snapshot,
        stdout_sha256=stdout_sha256,
        stderr_path=stderr_snapshot,
        stderr_sha256=stderr_sha256,
        backend_requested=backend_requested,
        backend_resolved=backend_resolved,
    )


def _percentile_r7(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise DriverError("percentile input is invalid")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0.0 for value in ordered):
        raise DriverError("latency observations must be positive finite values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "median_ms": _percentile_r7(values, 0.5),
        "p95_ms": _percentile_r7(values, 0.95),
        "p99_ms": _percentile_r7(values, 0.99),
    }


def _overlap_peak(rows: Sequence[Mapping[str, Any]], *, start_key: str, end_key: str) -> int:
    events: list[tuple[int, int]] = []
    for row in rows:
        start = row.get(start_key)
        end = row.get(end_key)
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise DriverError(f"raw request has invalid {start_key}/{end_key} interval")
        events.append((start, 1))
        events.append((end, -1))
    active = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        if active < 0:
            raise DriverError("raw request intervals are unbalanced")
        peak = max(peak, active)
    if active != 0:
        raise DriverError("raw request intervals are unbalanced")
    return peak


def _n06a_validate_client_row(row: Mapping[str, Any], workload: Workload) -> None:
    if row.get("status") != "success" or row.get("protocol_valid") is not True or row.get("reference_match") is not True:
        raise DriverError("strict streaming client did not complete the exact reference response")
    if row.get("streaming") is not True or row.get("mode") != "strict":
        raise DriverError("N06-A requires strict streaming token observations")
    if row.get("token_ids") != list(workload.output_token_ids):
        raise DriverError("strict streaming client output IDs differ from the fixed workload")
    if row.get("prompt_token_ids") != list(workload.prompt_token_ids):
        raise DriverError("strict streaming client prompt IDs differ from the fixed workload")
    groups = row.get("token_delivery_groups")
    if not isinstance(groups, dict) or groups.get("frame_token_counts") != [1] * workload.max_output_tokens:
        raise DriverError("N06-A requires exactly one generated token ID per SSE event")
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise DriverError("streaming client omitted metrics")
    for key in ("e2e_ns", "token_ttft_ns", "token_tpot_ns"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise DriverError(f"streaming client {key} must be positive and finite")
    token_arrival = row.get("token_arrival_ns")
    if not isinstance(token_arrival, list) or len(token_arrival) != workload.max_output_tokens:
        raise DriverError("streaming client did not retain one timestamp per output token")


def run_streaming_phase(
    *,
    port: int,
    workload: Workload,
    concurrency: int,
    count: int,
    request_timeout_seconds: float,
    phase: str = "retained",
    client_factory: Callable[[], Any] = token_client.TokenHttpClient,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Closed-loop C<=32 phase using the shared strict token SSE parser.

    ``serving_token_client_v2.run_phase`` deliberately caps its legacy contract
    at C<=8.  The HTTP client itself is thread-safe, so this bounded driver
    owns a C<=32 loop while preserving that client's raw response records and
    deadline watchdog.
    """
    if phase not in {"server-warmup", "retained"}:
        raise DriverError("streaming phase must be server-warmup or retained")
    if not 1 <= concurrency <= MAX_CONCURRENCY or count < concurrency:
        raise DriverError("streaming phase has invalid concurrency/count")
    reference = workload.reference()
    payload = {
        "model": MODEL_ID,
        "prompt": workload.prompt,
        "max_tokens": workload.max_output_tokens,
        "temperature": workload.temperature,
        "top_p": workload.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    lock = threading.Lock()
    stop = threading.Event()
    state: dict[str, Any] = {
        "next": 0,
        "rows": [],
        "errors": [],
        "started_ns": monotonic_ns(),
    }
    barrier = threading.Barrier(concurrency + 1, action=lambda: state.update(started_ns=monotonic_ns()))

    with client_factory() as client:
        def worker(worker_id: int) -> None:
            try:
                barrier.wait(timeout=30.0)
            except BaseException as error:
                with lock:
                    state["errors"].append({"type": type(error).__name__, "message": str(error), "stage": "barrier"})
                stop.set()
                return
            while not stop.is_set():
                with lock:
                    if stop.is_set() or state["next"] >= count:
                        return
                    index = state["next"]
                    state["next"] += 1
                call_started = monotonic_ns()
                try:
                    observed_row = client.request(
                        port,
                        payload,
                        reference,
                        streaming=True,
                        mode="strict",
                        timeout_seconds=request_timeout_seconds,
                    )
                    row = dict(observed_row)
                    try:
                        _n06a_validate_client_row(row, workload)
                    except BaseException as error:
                        # Keep the parser's complete raw transport/frame record even
                        # when the N06-A single-token/exact-output contract rejects it.
                        row["status"] = "failed"
                        row["n06a_validation_error"] = {"type": type(error).__name__, "message": str(error)}
                        stop.set()
                        with contextlib.suppress(Exception):
                            client.abort_pending("N06-A lane request failed")
                except BaseException as error:
                    finished = monotonic_ns()
                    row = {
                        "schema_version": token_client.SCHEMA,
                        "status": "failed",
                        "started_ns": call_started,
                        "finished_ns": finished,
                        "call_finished_ns": finished,
                        "error": {"type": type(error).__name__, "message": str(error)},
                        "token_ids": [],
                        "prompt_token_ids": None,
                        "protocol_valid": False,
                        "reference_match": False,
                        "token_delivery_groups": None,
                        "metrics": {},
                    }
                    stop.set()
                    with contextlib.suppress(Exception):
                        client.abort_pending("N06-A lane request failed")
                row = dict(row)
                row.update(
                    index=index,
                    worker_id=worker_id,
                    phase=phase,
                    warmup=phase != "retained",
                    call_started_ns=call_started,
                    call_finished_ns=max(monotonic_ns(), int(row.get("call_finished_ns", call_started))),
                )
                with lock:
                    state["rows"].append(row)

        executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="n06a-token")
        futures = [executor.submit(worker, worker_id) for worker_id in range(concurrency)]
        try:
            barrier.wait(timeout=30.0)
            for future in as_completed(futures):
                future.result()
        except BaseException as error:
            stop.set()
            with lock:
                state["errors"].append({"type": type(error).__name__, "message": str(error), "stage": "phase"})
            with contextlib.suppress(Exception):
                client.abort_pending("N06-A phase interrupted")
        finally:
            stop.set()
            barrier.abort()
            executor.shutdown(wait=True, cancel_futures=True)
    finished_ns = monotonic_ns()
    rows = sorted(state["rows"], key=lambda item: int(item["index"]))
    successful = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    response_ids = [
        row["response_identity"].get("id")
        for row in successful
        if isinstance(row.get("response_identity"), Mapping)
    ]
    duplicate_response_ids = sorted(
        response_id for response_id, amount in Counter(response_ids).items() if response_id is not None and amount > 1
    )
    observed = _overlap_peak(rows, start_key="started_ns", end_key="finished_ns") if rows else 0
    phase_wall_ns = finished_ns - int(state["started_ns"])
    if phase_wall_ns <= 0:
        raise DriverError("streaming phase wall clock did not advance")
    completed = (
        len(successful) == count
        and not failures
        and len(rows) == count
        and state["next"] == count
        and not state["errors"]
        and observed == concurrency
        and len(response_ids) == len(successful)
        and not duplicate_response_ids
    )
    accounting = {
        "schema_version": "riley.n06a-streaming-phase.v1",
        "phase": phase,
        "warmup": phase != "retained",
        "offered_concurrency": concurrency,
        "requested": count,
        "attempted": state["next"],
        "succeeded": len(successful),
        "failed": len(failures),
        "unresolved_attempts": state["next"] - len(rows),
        "not_started": count - state["next"],
        "observed_max_request_in_flight": observed,
        "phase_started_ns": state["started_ns"],
        "phase_finished_ns": finished_ns,
        "phase_wall_ns": phase_wall_ns,
        "duplicate_response_ids": duplicate_response_ids,
        "worker_errors": state["errors"],
        "completed": completed,
        "failure_policy": "stop refill, drain owned in-flight requests, preserve all started rows, no retries",
        "timing_scope": "client-observed single-token SSE delivery; not CUDA or scheduler commit time",
    }
    return rows, accounting


def _require_completed_server_warmup(
    *, lane: str, rows: Sequence[Mapping[str, Any]], accounting: Mapping[str, Any], expected_count: int, concurrency: int
) -> None:
    """Reject any partial/failed fresh-server warmup before retained timing."""
    expected = {
        "phase": "server-warmup",
        "warmup": True,
        "offered_concurrency": concurrency,
        "requested": expected_count,
        "attempted": expected_count,
        "succeeded": expected_count,
        "failed": 0,
        "completed": True,
    }
    for key, value in expected.items():
        if accounting.get(key) != value:
            raise DriverError(f"{lane} strict per-server warmup has invalid {key}: {accounting.get(key)!r}")
    if len(rows) != expected_count:
        raise DriverError(f"{lane} strict per-server warmup did not retain every request row")
    if any(row.get("status") != "success" for row in rows):
        raise DriverError(f"{lane} strict per-server warmup retained a failed request")


def _measurement_from_phase(lane: str, rows: Sequence[Mapping[str, Any]], accounting: Mapping[str, Any]) -> LaneMeasurement:
    if accounting.get("completed") is not True:
        raise DriverError(f"{lane} retained streaming phase did not complete without failures")
    if not rows or len(rows) != accounting.get("requested"):
        raise DriverError(f"{lane} retained streaming rows do not match requested count")
    output_tokens = sum(len(row.get("token_ids", [])) for row in rows)
    wall_ns = accounting.get("phase_wall_ns")
    if not isinstance(wall_ns, int) or wall_ns <= 0 or output_tokens <= 0:
        raise DriverError(f"{lane} retained throughput inputs are invalid")
    ttft = _distribution([float(row["metrics"]["token_ttft_ns"]) / 1_000_000.0 for row in rows])
    tpot = _distribution([float(row["metrics"]["token_tpot_ns"]) / 1_000_000.0 for row in rows])
    e2e = _distribution([float(row["metrics"]["e2e_ns"]) / 1_000_000.0 for row in rows])
    return LaneMeasurement(
        lane=lane,
        request_rows=tuple(rows),
        accounting=accounting,
        retained_wall_ms=wall_ns / 1_000_000.0,
        output_tokens_per_second=output_tokens * 1_000_000_000.0 / wall_ns,
        ttft=ttft,
        tpot=tpot,
        e2e=e2e,
        p99_status="qualified" if len(rows) >= 1_000 else "descriptive",
    )


def _default_stop_process(managed: ManagedProcess, timeout_seconds: float) -> Mapping[str, Any]:
    process = managed.process
    actions: list[str] = []
    errors: list[str] = []
    if process.poll() is None and managed.used_shutdown_stdin and getattr(process, "stdin", None) is not None:
        try:
            process.stdin.write(b"\n")
            process.stdin.flush()
            actions.append("stdin-newline")
        except Exception as error:
            errors.append(f"stdin: {type(error).__name__}: {error}")
    if process.poll() is None:
        try:
            process.wait(timeout=timeout_seconds)
            actions.append("wait")
        except subprocess.TimeoutExpired:
            pass
        except Exception as error:
            errors.append(f"wait: {type(error).__name__}: {error}")
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
                actions.append("process-group-sigterm")
            else:
                process.terminate()
                actions.append("terminate")
        except ProcessLookupError:
            pass
        except Exception as error:
            errors.append(f"terminate: {type(error).__name__}: {error}")
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                    actions.append("process-group-sigkill")
                else:
                    process.kill()
                    actions.append("kill")
            except ProcessLookupError:
                pass
            except Exception as error:
                errors.append(f"kill: {type(error).__name__}: {error}")
            try:
                process.wait(timeout=timeout_seconds)
            except Exception as error:
                errors.append(f"final-wait: {type(error).__name__}: {error}")
        except Exception as error:
            errors.append(f"term-wait: {type(error).__name__}: {error}")
    for stream in (managed.stdout_stream, managed.stderr_stream):
        with contextlib.suppress(Exception):
            stream.flush()
        with contextlib.suppress(Exception):
            stream.close()
    return {
        "lane": managed.lane,
        "pid": process.pid,
        "returncode": process.poll(),
        "actions": actions,
        "errors": errors,
        "cleanup_verified": process.poll() is not None and not errors,
    }


def _docker_cleanup_command(
    launcher: str, arguments: Sequence[str], *, timeout_seconds: float
) -> dict[str, Any]:
    """Run one bounded Docker control-plane operation without a shell.

    stdout/stderr are represented by lengths and digests in the receipt rather
    than copied into a marker.  The final container-list operation must still
    report an empty stdout payload to establish daemon-visible absence.
    """
    argv = [launcher, *arguments]
    command_timeout = min(MAX_DOCKER_CLEANUP_COMMAND_SECONDS, max(1.0, timeout_seconds))
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            timeout=command_timeout,
            shell=False,
            check=False,
        )
    except BaseException as error:
        return {
            "argv": argv,
            "timeout_seconds": command_timeout,
            "returncode": None,
            "stdout_bytes": 0,
            "stdout_sha256": _sha256(b""),
            "stderr_bytes": 0,
            "stderr_sha256": _sha256(b""),
            "error": f"{type(error).__name__}: {error}",
        }
    stdout, stderr = completed.stdout, completed.stderr
    result: dict[str, Any] = {
        "argv": argv,
        "timeout_seconds": command_timeout,
        "returncode": completed.returncode,
        "stdout_bytes": len(stdout),
        "stdout_sha256": _sha256(stdout),
        "stderr_bytes": len(stderr),
        "stderr_sha256": _sha256(stderr),
    }
    if len(stdout) > MAX_DOCKER_CLEANUP_OUTPUT_BYTES or len(stderr) > MAX_DOCKER_CLEANUP_OUTPUT_BYTES:
        result["error"] = (
            "Docker cleanup command output exceeds "
            f"{MAX_DOCKER_CLEANUP_OUTPUT_BYTES} bytes"
        )
        return result
    try:
        result["stdout_text"] = stdout.decode("utf-8")
        result["stderr_text"] = stderr.decode("utf-8")
    except UnicodeDecodeError as error:
        result["error"] = f"Docker cleanup command output is not UTF-8: {error}"
    return result


def _docker_container_inventory(
    launcher: str, container_name: str, *, timeout_seconds: float
) -> tuple[dict[str, Any], bool | None, str | None]:
    """Use a daemon-backed exact-name inventory to distinguish absent from unavailable."""
    command = _docker_cleanup_command(
        launcher,
        (
            "container",
            "ls",
            "--all",
            "--filter",
            f"name=^/{container_name}$",
            "--format",
            "{{.Names}}\\t{{.ID}}",
        ),
        timeout_seconds=timeout_seconds,
    )
    if command.get("error") is not None:
        return command, None, str(command["error"])
    if command.get("returncode") != 0:
        return command, None, "Docker container inventory exited non-zero"
    text = command.get("stdout_text")
    if not isinstance(text, str):
        return command, None, "Docker container inventory did not retain UTF-8 stdout"
    lines = [line for line in text.splitlines() if line]
    if not lines:
        return command, False, None
    if len(lines) != 1:
        return command, None, "Docker exact-name inventory returned multiple containers"
    name, separator, container_id = lines[0].partition("\t")
    if separator != "\t" or name != container_name or not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
        return command, None, "Docker exact-name inventory has an unexpected row"
    return command, True, None


def _default_cleanup_vllm_container(
    launcher: str, container_name: str, timeout_seconds: float
) -> Mapping[str, Any]:
    """Prove that the owned non-detached vLLM Docker container is gone.

    Process-group termination only proves the local Docker CLI ended.  This
    receipt queries the daemon by the deterministic name, attempts stop/wait
    when still present, force-removes a stubborn residue, then requires both a
    successful exact-name inventory and a non-successful `container inspect`.
    The latter alone cannot distinguish a missing daemon from a missing
    container; the successful list query supplies that distinction.
    """
    receipt: dict[str, Any] = {
        "schema_version": DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION,
        "launcher": launcher,
        "container_name": container_name,
        "commands": [],
        "notes": [],
        "errors": [],
        "cleanup_verified": False,
    }
    if not DOCKER_CONTAINER_NAME_RE.fullmatch(container_name):
        receipt["errors"].append("derived Docker container name is invalid")
        return receipt
    before, present, inventory_error = _docker_container_inventory(
        launcher, container_name, timeout_seconds=timeout_seconds
    )
    receipt["commands"].append({"operation": "inventory-before", **before})
    if inventory_error is not None:
        receipt["errors"].append(inventory_error)
        return receipt
    if present:
        stop = _docker_cleanup_command(
            launcher,
            ("container", "stop", "--time", str(max(1, math.ceil(min(timeout_seconds, MAX_DOCKER_CLEANUP_COMMAND_SECONDS)))), container_name),
            timeout_seconds=timeout_seconds,
        )
        receipt["commands"].append({"operation": "stop", **stop})
        if stop.get("error") is not None:
            receipt["errors"].append(str(stop["error"]))
            return receipt
        if stop.get("returncode") != 0:
            receipt["notes"].append("Docker stop returned non-zero; final daemon absence remains required")
        wait = _docker_cleanup_command(
            launcher,
            ("container", "wait", container_name),
            timeout_seconds=timeout_seconds,
        )
        receipt["commands"].append({"operation": "wait", **wait})
        if wait.get("error") is not None:
            receipt["errors"].append(str(wait["error"]))
            return receipt
        if wait.get("returncode") != 0:
            receipt["notes"].append("Docker wait returned non-zero after stop; final daemon absence remains required")

    after_stop, still_present, inventory_error = _docker_container_inventory(
        launcher, container_name, timeout_seconds=timeout_seconds
    )
    receipt["commands"].append({"operation": "inventory-after-stop", **after_stop})
    if inventory_error is not None:
        receipt["errors"].append(inventory_error)
        return receipt
    if still_present:
        remove = _docker_cleanup_command(
            launcher,
            ("container", "rm", "--force", container_name),
            timeout_seconds=timeout_seconds,
        )
        receipt["commands"].append({"operation": "force-remove", **remove})
        if remove.get("error") is not None or remove.get("returncode") != 0:
            receipt["errors"].append(
                str(remove.get("error") or "Docker force-remove exited non-zero")
            )
            return receipt
        final_inventory, final_present, inventory_error = _docker_container_inventory(
            launcher, container_name, timeout_seconds=timeout_seconds
        )
        receipt["commands"].append({"operation": "inventory-after-force-remove", **final_inventory})
    else:
        final_inventory, final_present, inventory_error = after_stop, still_present, None
    if inventory_error is not None or final_present is not False:
        receipt["errors"].append(inventory_error or "Docker container remains present after cleanup")
        return receipt
    inspect = _docker_cleanup_command(
        launcher,
        ("container", "inspect", "--format", "{{.Id}}", container_name),
        timeout_seconds=timeout_seconds,
    )
    receipt["commands"].append({"operation": "inspect-absence", **inspect})
    if inspect.get("error") is not None:
        receipt["errors"].append(str(inspect["error"]))
        return receipt
    if inspect.get("returncode") == 0:
        receipt["errors"].append("Docker inspect still resolves the owned container after absence inventory")
        return receipt
    receipt["final_state"] = "absent"
    receipt["cleanup_verified"] = True
    return receipt


def _launch_version_probe(config: DriverConfig, attempt_dir: Path) -> Mapping[str, Any] | None:
    if config.vllm_version_template is None:
        return None
    argv = _expand_argv(
        config.vllm_version_template,
        {
            "port": str(config.vllm_port),
            "model_path": str(config.model_path),
            "model_id": MODEL_ID,
            "concurrency": str(config.concurrency),
            "batch_token_budget": str(config.batch_token_budget),
            "max_model_len": str(config.max_model_len),
            "max_output_tokens": str(config.workload.max_output_tokens),
            "kv_blocks": str(config.kv_blocks),
        },
        label="vllm-version-command-json",
    )
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(attempt_dir),
            env=_child_environment(config),
            capture_output=True,
            timeout=30.0,
            shell=False,
            check=False,
        )
    except Exception as error:
        return {"argv": list(argv), "status": "unavailable", "error": f"{type(error).__name__}: {error}"}
    stdout_path = attempt_dir / "vllm.version.stdout.log"
    stderr_path = attempt_dir / "vllm.version.stderr.log"
    _write_bytes_create_only(stdout_path, completed.stdout)
    _write_bytes_create_only(stderr_path, completed.stderr)
    return {
        "argv": list(argv),
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout_path": str(stdout_path.resolve(strict=True)),
        "stdout_sha256": _sha256(completed.stdout),
        "stderr_path": str(stderr_path.resolve(strict=True)),
        "stderr_sha256": _sha256(completed.stderr),
    }


def _marker_values(
    *,
    measurement: LaneMeasurement,
    config: DriverConfig,
    context: AttemptContext,
    startup_snapshot: StartupSnapshot | None,
    vllm_startup_snapshot: VllmStartupSnapshot | None,
    retained_artifacts: RetainedArtifacts,
    server_warmup_artifacts: ServerWarmupArtifacts,
    model_identity_snapshot: tuple[Path, str],
    model_identity_validation: tuple[Path, str],
    lane_provenance: LaneProvenanceArtifact,
    whole_gpu_memory: Mapping[str, Any],
) -> dict[str, str]:
    workload = config.workload
    required_gpu_evidence = {
        "sample_path",
        "sample_sha256",
        "peak_path",
        "peak_sha256",
        "sample_count",
        "peak_used_bytes",
        "peak_limit_bytes",
        "compliant",
    }
    if not required_gpu_evidence <= set(whole_gpu_memory) or whole_gpu_memory.get("compliant") is not True:
        raise DriverError("marker requires a compliant whole-GPU sampled peak receipt")
    if whole_gpu_memory["peak_limit_bytes"] != config.whole_gpu_sampled_peak_limit_bytes:
        raise DriverError("whole-GPU sampled peak receipt limit differs from the driver configuration")
    values = {
        "schema_version": "1",
        "lane": measurement.lane,
        "case": workload.case,
        "pair_order": context.pair_order,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_tokens": str(workload.prompt_tokens),
        "max_output_tokens": str(workload.max_output_tokens),
        "offered_concurrency": str(config.concurrency),
        "batch_token_budget": str(config.batch_token_budget),
        "max_model_len": str(config.max_model_len),
        "model_identity_manifest_path": str(model_identity_snapshot[0]),
        "model_identity_manifest_sha256": model_identity_snapshot[1],
        "model_identity_validation_path": str(model_identity_validation[0]),
        "model_identity_validation_sha256": model_identity_validation[1],
        "vllm_prefix_caching": "disabled",
        "vllm_image_digest": config.vllm_image_digest,
        "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
        "whole_gpu_sampled_peak_limit_bytes": str(config.whole_gpu_sampled_peak_limit_bytes),
        "whole_gpu_sampled_peak_bytes": str(whole_gpu_memory["peak_used_bytes"]),
        "whole_gpu_sample_path": str(whole_gpu_memory["sample_path"]),
        "whole_gpu_sample_sha256": str(whole_gpu_memory["sample_sha256"]),
        "whole_gpu_peak_receipt_path": str(whole_gpu_memory["peak_path"]),
        "whole_gpu_peak_receipt_sha256": str(whole_gpu_memory["peak_sha256"]),
        "request_rows_path": str(retained_artifacts.request_rows_path),
        "request_rows_sha256": retained_artifacts.request_rows_sha256,
        "phase_path": str(retained_artifacts.phase_path),
        "phase_sha256": retained_artifacts.phase_sha256,
        "server_warmup_request_rows_path": str(server_warmup_artifacts.request_rows_path),
        "server_warmup_request_rows_sha256": server_warmup_artifacts.request_rows_sha256,
        "server_warmup_phase_path": str(server_warmup_artifacts.phase_path),
        "server_warmup_phase_sha256": server_warmup_artifacts.phase_sha256,
        "attempt_config_path": str(retained_artifacts.attempt_config_path),
        "attempt_config_sha256": retained_artifacts.attempt_config_sha256,
        "lane_provenance_path": str(lane_provenance.path),
        "lane_provenance_sha256": lane_provenance.sha256,
        "attempted_requests": str(measurement.accounting["attempted"]),
        "successful_requests": str(measurement.accounting["succeeded"]),
        "failed_requests": str(measurement.accounting["failed"]),
        "output_tokens": str(measurement.accounting["succeeded"] * workload.max_output_tokens),
        "retained_wall_ms": f"{measurement.retained_wall_ms:.6f}",
        "output_tokens_per_second": f"{measurement.output_tokens_per_second:.6f}",
        "token_timing_scope": "client-observed-single-token-sse",
        "token_ttft_median_ms": f"{measurement.ttft['median_ms']:.6f}",
        "token_ttft_p95_ms": f"{measurement.ttft['p95_ms']:.6f}",
        "token_ttft_p99_ms": f"{measurement.ttft['p99_ms']:.6f}",
        "token_tpot_median_ms": f"{measurement.tpot['median_ms']:.6f}",
        "token_tpot_p95_ms": f"{measurement.tpot['p95_ms']:.6f}",
        "token_tpot_p99_ms": f"{measurement.tpot['p99_ms']:.6f}",
        "e2e_median_ms": f"{measurement.e2e['median_ms']:.6f}",
        "e2e_p95_ms": f"{measurement.e2e['p95_ms']:.6f}",
        "e2e_p99_ms": f"{measurement.e2e['p99_ms']:.6f}",
        "latency_sample_count": str(measurement.accounting["succeeded"]),
        "p99_status": measurement.p99_status,
        "quality_status": "passed",
        "status": "passed",
    }
    if measurement.lane == "riley":
        if startup_snapshot is None:
            raise DriverError("Riley marker requires its startup stderr snapshot")
        values.update(
            {
                "backend_requested": REQUESTED_BACKEND_CLI_ID,
                "backend_resolved": RESOLVED_BACKEND_ID,
                "fallback_reason": "none",
                "startup_log_path": str(startup_snapshot.path),
                "startup_log_sha256": startup_snapshot.sha256,
                "query_heads": "16",
                "key_value_heads": "2",
                "head_size": "128",
                "page_size": "16",
                "graph_capture_enabled": "false",
            }
        )
    else:
        if vllm_startup_snapshot is None:
            raise DriverError("vLLM marker requires one backend-attested startup snapshot")
        values.update(
            {
                "backend_requested": vllm_startup_snapshot.backend_requested,
                "backend_resolved": vllm_startup_snapshot.backend_resolved,
                "fallback_reason": "none",
                "startup_stdout_log_path": str(vllm_startup_snapshot.stdout_path),
                "startup_stdout_log_sha256": vllm_startup_snapshot.stdout_sha256,
                "startup_stderr_log_path": str(vllm_startup_snapshot.stderr_path),
                "startup_stderr_log_sha256": vllm_startup_snapshot.stderr_sha256,
            }
        )
    return values


def render_marker(values: Mapping[str, str]) -> str:
    if any(not key or not value or any(character.isspace() for character in value) for key, value in values.items()):
        raise DriverError("marker contains an empty or whitespace-bearing field")
    return MARKER_PREFIX + " " + " ".join(f"{key}={value}" for key, value in values.items())


def _lane_provenance(
    managed: ManagedProcess,
    ready: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    whole_gpu_memory: Mapping[str, Any],
    cleanup_completed_ns: int,
) -> dict[str, Any]:
    return {
        "argv": list(managed.command),
        "pid": managed.process.pid,
        "started_ns": managed.started_ns,
        "ready": dict(ready),
        "stdout_path": str(managed.stdout_path.resolve(strict=True)),
        "stderr_path": str(managed.stderr_path.resolve(strict=True)),
        "cleanup": dict(cleanup),
        "cleanup_completed_ns": cleanup_completed_ns,
        "whole_gpu_memory": dict(whole_gpu_memory),
    }


def _run_lane(
    *,
    lane: str,
    config: DriverConfig,
    context: AttemptContext,
    attempt_dir: Path,
    dependencies: DriverDependencies,
    request_count: int,
    retain_measurement: bool,
    vllm_container_name: str | None,
) -> tuple[
    LaneMeasurement | None,
    StartupSnapshot | None,
    VllmStartupSnapshot | None,
    Mapping[str, Any],
    LaneProvenanceArtifact,
]:
    if lane == "riley":
        argv = _riley_argv(config)
    else:
        if vllm_container_name is None:
            raise DriverError("vLLM lane requires a deterministic owned Docker container name")
        argv = _vllm_argv_for_container(config, vllm_container_name)
    process: ManagedProcess | None = None
    ready: Mapping[str, Any] = {}
    cleanup: Mapping[str, Any] = {}
    snapshot: StartupSnapshot | None = None
    vllm_snapshot: VllmStartupSnapshot | None = None
    measurement: LaneMeasurement | None = None
    failure: BaseException | None = None
    gpu_evidence: dict[str, Any] = {}
    provenance_artifact: LaneProvenanceArtifact | None = None
    cleanup_completed_ns: int | None = None
    post_lane_gpu_idle: Mapping[str, Any] = {}
    sampler_factory = dependencies.gpu_memory_sampler_factory or WholeGpuMemorySampler
    sampler = sampler_factory(
        nvidia_smi=config.nvidia_smi,
        interval_seconds=config.gpu_memory_sampling_interval_seconds,
        monotonic_ns=dependencies.monotonic_ns,
    )
    sampler_started = False
    try:
        # The initial physical-GPU sample must precede Popen so an allocation
        # spike during server construction cannot fall in an unobserved gap.
        sampler.start()
        sampler_started = True
        process = _launch_server(
            lane=lane,
            argv=argv,
            attempt_dir=attempt_dir,
            dependencies=dependencies,
            environment=_child_environment(config),
            use_shutdown_stdin=lane == "riley",
        )
        waiter = dependencies.wait_ready or wait_ready
        ready_result = waiter(
            process.process,
            config.riley_port if lane == "riley" else config.vllm_port,
            lane,
            config.startup_timeout_seconds,
        )
        if not isinstance(ready_result, Mapping):
            raise DriverError(f"{lane} readiness receipt must be an object")
        # Capture the controller-side readiness boundary.  A readiness helper
        # may report endpoint/status details, but it cannot supply a trusted
        # lifecycle clock for the retained phase.
        ready = {**dict(ready_result), "ready_ns": dependencies.monotonic_ns()}
        if lane == "riley":
            snapshot = capture_riley_startup_snapshot(process.stderr_path, attempt_dir / "riley.startup.stderr.log")
        else:
            vllm_snapshot = capture_vllm_startup_snapshot(
                stdout_path=process.stdout_path,
                stderr_path=process.stderr_path,
                stdout_destination=attempt_dir / "vllm.startup.stdout.log",
                stderr_destination=attempt_dir / "vllm.startup.stderr.log",
                backend_requested=config.vllm_backend_requested,
                backend_receipt_regex=config.vllm_backend_receipt_regex,
                required_fragments=config.vllm_startup_required_fragments,
            )
        runner = dependencies.run_streaming_phase or run_streaming_phase
        common = {
            "port": config.riley_port if lane == "riley" else config.vllm_port,
            "workload": config.workload,
            "concurrency": config.concurrency,
            "request_timeout_seconds": config.request_timeout_seconds,
            "monotonic_ns": dependencies.monotonic_ns,
        }
        warmup_rows, warmup_accounting = runner(
            **common,
            count=config.warmup_requests,
            phase="server-warmup",
        )
        _write_jsonl_create_only(attempt_dir / f"{lane}.server-warmup.requests.jsonl", warmup_rows)
        _write_json_create_only(attempt_dir / f"{lane}.server-warmup.phase.json", dict(warmup_accounting))
        _require_completed_server_warmup(
            lane=lane,
            rows=warmup_rows,
            accounting=warmup_accounting,
            expected_count=config.warmup_requests,
            concurrency=config.concurrency,
        )
        if retain_measurement:
            rows, accounting = runner(
                **common,
                count=request_count,
                phase="retained",
            )
            _write_jsonl_create_only(attempt_dir / f"{lane}.requests.jsonl", rows)
            _write_json_create_only(attempt_dir / f"{lane}.phase.json", dict(accounting))
            measurement = _measurement_from_phase(lane, rows, accounting)
    except BaseException as error:
        failure = error
    finally:
        cleanup_error: DriverError | None = None
        gpu_error: DriverError | None = None
        if process is not None:
            stopper = dependencies.stop_process or _default_stop_process
            try:
                cleanup = stopper(process, config.shutdown_timeout_seconds)
            except BaseException as error:
                cleanup = {
                    "lane": lane,
                    "pid": process.process.pid,
                    "cleanup_verified": False,
                    "errors": [f"{type(error).__name__}: {error}"],
                }
                if failure is None:
                    cleanup_error = DriverError(f"{lane} cleanup failed: {type(error).__name__}: {error}")
            if not cleanup.get("cleanup_verified") and failure is None:
                cleanup_error = DriverError(f"{lane} cleanup did not prove owned process exit")
            if lane == "vllm":
                if vllm_container_name is None:
                    raise DriverError("vLLM cleanup lost its deterministic Docker container name")
                cleaner = dependencies.cleanup_vllm_container or _default_cleanup_vllm_container
                try:
                    container_cleanup = cleaner(
                        str(config.vllm_template[0]),
                        vllm_container_name,
                        config.shutdown_timeout_seconds,
                    )
                    cleanup = {**cleanup, "container": dict(container_cleanup)}
                except BaseException as error:
                    cleanup = {
                        **cleanup,
                        "container": {
                            "schema_version": DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION,
                            "container_name": vllm_container_name,
                            "cleanup_verified": False,
                            "errors": [f"{type(error).__name__}: {error}"],
                        },
                    }
                    if failure is None:
                        cleanup_error = DriverError(
                            f"vLLM Docker container cleanup failed: {type(error).__name__}: {error}"
                        )
                container = cleanup.get("container")
                if not isinstance(container, Mapping) or container.get("cleanup_verified") is not True:
                    if failure is None:
                        cleanup_error = DriverError(
                            "vLLM Docker cleanup did not prove the owned daemon container is absent"
                        )
            cleanup_completed_ns = dependencies.monotonic_ns()
        if sampler_started:
            try:
                sample_result = sampler.stop()
                if process is None or cleanup_completed_ns is None:
                    raise DriverError("owned server did not launch, so whole-GPU lifetime evidence is unavailable")
                gpu_evidence.update(
                    _write_whole_gpu_memory_evidence(
                        attempt_dir=attempt_dir,
                        lane=lane,
                        sample_result=sample_result,
                        peak_limit_bytes=config.whole_gpu_sampled_peak_limit_bytes,
                        expected_interval_seconds=config.gpu_memory_sampling_interval_seconds,
                        server_started_ns=process.started_ns,
                        server_cleanup_completed_ns=cleanup_completed_ns,
                    )
                )
                if not gpu_evidence["compliant"]:
                    detail = (
                        f"samples={gpu_evidence['sample_count']} peak={gpu_evidence['peak_used_bytes']} "
                        f"limit={gpu_evidence['peak_limit_bytes']} errors={gpu_evidence['errors']}"
                    )
                    if failure is None:
                        gpu_error = DriverError(f"{lane} whole-GPU sampled peak is not compliant: {detail}")
                    gpu_evidence["compliance_error"] = detail
            except BaseException as error:
                if failure is None:
                    gpu_error = DriverError(f"{lane} whole-GPU sampling failed: {type(error).__name__}: {error}")
                gpu_evidence["sampling_error"] = f"{type(error).__name__}: {error}"
        if process is not None:
            idle_error: DriverError | None = None
            census = dependencies.gpu_idle_census or _default_gpu_idle_census
            try:
                post_lane_gpu_idle = census(
                    nvidia_smi=config.nvidia_smi,
                    timeout_seconds=GPU_IDLE_CENSUS_COMMAND_TIMEOUT_SECONDS,
                    monotonic_ns=dependencies.monotonic_ns,
                )
                _validate_post_lane_gpu_idle_census(
                    post_lane_gpu_idle,
                    nvidia_smi=config.nvidia_smi,
                    label=lane,
                )
            except BaseException as error:
                post_lane_gpu_idle = {
                    "schema_version": GPU_IDLE_CENSUS_SCHEMA_VERSION,
                    "validation_error": f"{type(error).__name__}: {error}",
                }
                if failure is None:
                    idle_error = DriverError(
                        f"{lane} post-lane GPU-0 idle census failed: {type(error).__name__}: {error}"
                    )
            if idle_error is not None and gpu_error is None:
                gpu_error = idle_error
        if process is not None:
            provenance = _lane_provenance(
                process,
                ready,
                cleanup,
                gpu_evidence,
                cleanup_completed_ns if cleanup_completed_ns is not None else process.started_ns,
            )
            provenance["post_lane_gpu_idle"] = dict(post_lane_gpu_idle)
            if snapshot is not None:
                provenance["riley_startup_snapshot"] = {
                    "path": str(snapshot.path),
                    "sha256": snapshot.sha256,
                    "receipt": dict(snapshot.receipt),
                }
            if vllm_snapshot is not None:
                provenance["vllm_startup_snapshot"] = {
                    "stdout_path": str(vllm_snapshot.stdout_path),
                    "stdout_sha256": vllm_snapshot.stdout_sha256,
                    "stderr_path": str(vllm_snapshot.stderr_path),
                    "stderr_sha256": vllm_snapshot.stderr_sha256,
                    "backend_requested": vllm_snapshot.backend_requested,
                    "backend_resolved": vllm_snapshot.backend_resolved,
                }
            provenance_path = attempt_dir / f"{lane}.provenance.json"
            _write_json_create_only(provenance_path, provenance)
            provenance_resolved, _, provenance_sha256 = _sha256_file(
                provenance_path,
                maximum_bytes=MAX_WORKLOAD_BYTES,
                label=f"{lane} lane provenance",
            )
            provenance_artifact = LaneProvenanceArtifact(provenance_resolved, provenance_sha256)
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    if cleanup_error is not None:
        raise cleanup_error
    if gpu_error is not None:
        raise gpu_error
    if process is None or provenance_artifact is None:
        raise DriverError(f"{lane} did not retain a launch/cleanup provenance artifact")
    if retain_measurement and measurement is None:
        raise DriverError(f"{lane} timed phase did not retain a measurement")
    return measurement, snapshot, vllm_snapshot, gpu_evidence, provenance_artifact


def execute_pair(
    config: DriverConfig,
    context: AttemptContext,
    *,
    dependencies: DriverDependencies | None = None,
) -> tuple[Path, tuple[str, ...]]:
    """Execute one outer N01 attempt and return its create-only evidence + markers.

    For a warmup, artifacts are retained but markers are deliberately empty.
    A timed call returns exactly the two order-bound marker lines only after both
    lane measurements and their process cleanups succeeded.
    """
    dependencies = dependencies or DriverDependencies()
    attempt_dir = _create_attempt_directory(config, context)
    pinned_workload_path, pinned_workload_sha256 = _freeze_workload_snapshot(config, attempt_dir)
    pinned_model_identity_path, pinned_model_identity_sha256 = _freeze_model_identity_snapshot(config, attempt_dir)
    model_identity_validation_path, model_identity_validation_sha256 = _write_model_identity_validation(
        config=config,
        attempt_dir=attempt_dir,
        dependencies=dependencies,
    )
    vllm_container_name = _vllm_container_name(attempt_dir, context)
    vllm_argv = _vllm_argv_for_container(config, vllm_container_name)
    version_probe = _launch_version_probe(config, attempt_dir)
    _write_json_create_only(
        attempt_dir / "attempt.config.json",
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "phase": context.phase,
            "index": context.index,
            "pair_order": context.pair_order if context.phase == "timed" else None,
            "workload": {
                "path": str(pinned_workload_path),
                "sha256": pinned_workload_sha256,
                "case": config.workload.case,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "prompt_tokens": config.workload.prompt_tokens,
                "max_output_tokens": config.workload.max_output_tokens,
                "reference_sha256": config.workload.reference().sha256,
            },
            "configuration": {
                "concurrency": config.concurrency,
                "batch_token_budget": config.batch_token_budget,
                "retained_requests": config.retained_requests,
                "warmup_requests": config.warmup_requests,
                "max_model_len": config.max_model_len,
                "kv_blocks": config.kv_blocks,
                "riley_port": config.riley_port,
                "vllm_port": config.vllm_port,
                "cuda_visible_devices": config.cuda_visible_devices,
                "whole_gpu_memory_sampling": {
                    "nvidia_smi": str(config.nvidia_smi),
                    "gpu_index": 0,
                    "sampling_interval_seconds": config.gpu_memory_sampling_interval_seconds,
                    "max_sample_duration_seconds": MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS,
                    "max_start_gap_seconds": (
                        config.gpu_memory_sampling_interval_seconds
                        + GPU_MEMORY_SAMPLE_SCHEDULING_SLACK_SECONDS
                    ),
                    "peak_limit_bytes": config.whole_gpu_sampled_peak_limit_bytes,
                },
                "startup_timeout_seconds": config.startup_timeout_seconds,
                "request_timeout_seconds": config.request_timeout_seconds,
                "shutdown_timeout_seconds": config.shutdown_timeout_seconds,
            },
            "launch_provenance": {
                "model_identity": {
                    "path": str(pinned_model_identity_path),
                    "sha256": pinned_model_identity_sha256,
                    "source_path": str(config.model_identity.source_path),
                    "source_sha256": config.model_identity.source_sha256,
                    "validation_path": str(model_identity_validation_path),
                    "validation_sha256": model_identity_validation_sha256,
                    "git_path": str(config.model_git),
                    "git_sha256": config.model_git_sha256,
                },
                "riley_binary": {"path": str(config.riley_binary), "sha256": config.riley_binary_sha256},
                "vllm_command_template": {"path": str(config.vllm_command_path), "sha256": config.vllm_command_sha256},
                "vllm_host_launcher": {"path": str(Path(config.vllm_template[0]).resolve()), "sha256": config.vllm_launcher_sha256},
                "vllm_image_digest": config.vllm_image_digest,
                "vllm_docker_container": {
                    "name": vllm_container_name,
                    "detached": False,
                    "cleanup_schema_version": DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION,
                },
                "vllm_backend_attestation": {
                    "backend_requested": config.vllm_backend_requested,
                    "startup_receipt_regex": config.vllm_backend_receipt_regex.pattern,
                    "startup_required_fragments": list(config.vllm_startup_required_fragments),
                    "required_match_count": 1,
                },
                "vllm_fixed_prompt_fairness": {
                    "prefix_caching": "disabled",
                    "disable_flag": VLLM_PREFIX_CACHING_DISABLE_FLAG,
                    "docker_gpu_binding": "device=0",
                    "gpu_memory_utilization": config.vllm_gpu_memory_utilization,
                    "whole_gpu_sampled_peak_limit_bytes": config.whole_gpu_sampled_peak_limit_bytes,
                },
            },
            "riley_argv": list(_riley_argv(config)),
            "vllm_argv": list(vllm_argv),
            "vllm_version_probe": version_probe,
            "controller": {
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": _sha256(Path(__file__).read_bytes()),
                "token_client_path": str(Path(token_client.__file__).resolve()),
                "token_client_sha256": _sha256(Path(token_client.__file__).read_bytes()),
                "python": sys.executable,
                "python_version": sys.version,
            },
            "limitations": [
                "This external Python driver does not enter Riley's Rust/native/CUDA serving runtime.",
                "A marker is emitted only after the exact fixed response and clean owned-process shutdown succeed for both lanes.",
                "Client token timestamps are delivery observations, not CUDA or scheduler-commit timestamps.",
                "The vLLM marker's backend_resolved field is derived from exactly one configured startup-log regex capture; its full engine/graph/KV interpretation remains reviewable raw provenance.",
            ],
        },
    )
    lanes = context.pair_order.split("-") if context.phase == "timed" else ["riley", "vllm"]
    request_count = config.retained_requests if context.phase == "timed" else config.warmup_requests
    emitted: list[str] = []
    try:
        for lane in lanes:
            measurement, snapshot, vllm_snapshot, whole_gpu_memory, lane_provenance = _run_lane(
                lane=lane,
                config=config,
                context=context,
                attempt_dir=attempt_dir,
                dependencies=dependencies,
                request_count=request_count,
                retain_measurement=context.phase == "timed",
                vllm_container_name=vllm_container_name,
            )
            if context.phase == "timed":
                if measurement is None:
                    raise DriverError("timed lane did not produce a retained measurement")
                marker = render_marker(
                    _marker_values(
                        measurement=measurement,
                        config=config,
                        context=context,
                        startup_snapshot=snapshot,
                        vllm_startup_snapshot=vllm_snapshot,
                        retained_artifacts=_retained_artifacts(attempt_dir, lane),
                        server_warmup_artifacts=_server_warmup_artifacts(attempt_dir, lane),
                        model_identity_snapshot=(pinned_model_identity_path, pinned_model_identity_sha256),
                        model_identity_validation=(model_identity_validation_path, model_identity_validation_sha256),
                        lane_provenance=lane_provenance,
                        whole_gpu_memory=whole_gpu_memory,
                    )
                )
                emitted.append(marker)
        if context.phase == "timed":
            if len(emitted) != 2:
                raise DriverError("timed attempt did not produce exactly two lane markers")
            _write_json_create_only(
                attempt_dir / "attempt.complete.json",
                {
                    "completed": True,
                    "phase": context.phase,
                    "index": context.index,
                    "pair_order": context.pair_order,
                    "markers": emitted,
                    "lanes": list(lanes),
                },
            )
        else:
            _write_json_create_only(
                attempt_dir / "attempt.complete.json",
                {
                    "completed": True,
                    "phase": context.phase,
                    "index": context.index,
                    "markers": [],
                    "lanes": list(lanes),
                },
            )
        return attempt_dir, tuple(emitted)
    except BaseException as error:
        _write_json_create_only(
            attempt_dir / "attempt.failure.json",
            {
                "completed": False,
                "phase": context.phase,
                "index": context.index,
                "emitted_markers": emitted,
                "error": {"type": type(error).__name__, "message": str(error)},
                "traceback": traceback.format_exc(),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path, help="create-only N06-A raw artifact root outside the checkout")
    parser.add_argument("--workload", required=True, type=Path, help="immutable exact Qwen3B JSON workload")
    parser.add_argument("--model-path", required=True, type=Path, help="local pinned Qwen2.5-3B checkpoint directory")
    parser.add_argument(
        "--model-identity-manifest",
        required=True,
        type=Path,
        help="bounded manifest that pins model revision, small-file hashes, and large-shard LFS OIDs/sizes",
    )
    parser.add_argument(
        "--model-git",
        type=Path,
        default=Path("/usr/bin/git"),
        help="absolute git executable used for bounded rev-parse and git-lfs metadata validation",
    )
    parser.add_argument("--riley-binary", required=True, type=Path, help="absolute CUDA-enabled riley executable")
    parser.add_argument("--vllm-command-json", required=True, type=Path, help="JSON argv template; use {port}, {model_path}, and {model_id} exactly once")
    parser.add_argument(
        "--vllm-image-digest",
        required=True,
        help="immutable sha256:<64-hex> digest embedded once in the Docker image argv",
    )
    parser.add_argument("--vllm-version-command-json", type=Path, help="optional JSON argv template for a bounded vLLM version/image-digest probe")
    parser.add_argument(
        "--vllm-backend-requested",
        default=VLLM_AUTO_BACKEND_REQUESTED,
        help="must identify the unforced vLLM auto attention selector retained with the log-derived backend",
    )
    parser.add_argument("--vllm-backend-receipt-regex", required=True, help="regex over the readiness-time vLLM stdout+stderr snapshot with exactly one named group (?P<backend_resolved>...)")
    parser.add_argument("--vllm-startup-required-fragment", action="append", default=[], help="repeat for each literal graph/compile/KV/backend startup-log fragment that must appear before a marker is eligible")
    parser.add_argument("--riley-port", type=int, default=18080)
    parser.add_argument("--vllm-port", type=int, default=18081)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--batch-token-budget", type=int, default=32, help="native D128 CUDA rows per iteration; must be >= concurrency and <= 32")
    parser.add_argument(
        "--retained-requests",
        type=int,
        default=1_000,
        help="per-lane retained requests; P99 is only qualified at 1000 or more",
    )
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--kv-blocks", type=int, default=4352, help="must fit C * ceil((prompt + output) / 16)")
    parser.add_argument("--startup-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--nvidia-smi",
        type=Path,
        default=Path("/usr/bin/nvidia-smi"),
        help="absolute host nvidia-smi used for 250-500ms whole-GPU peak samples",
    )
    parser.add_argument(
        "--gpu-memory-sampling-interval-seconds",
        type=float,
        default=GPU_MEMORY_SAMPLE_INTERVAL_SECONDS,
        help="whole-GPU polling interval, bounded to 0.25..0.5 seconds",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _interrupt_for_owned_cleanup(_: int, __: object) -> None:
        # N01 sends SIGTERM before SIGKILL.  Convert it to the same control
        # path as Ctrl-C so execute_pair's lane finally blocks can close the
        # direct Riley/Docker child sessions and retain their receipts.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _interrupt_for_owned_cleanup)
    try:
        args = parser.parse_args(argv)
        context = attempt_context_from_environment()
        config = build_config(args)
        _, markers = execute_pair(config, context)
        for marker in markers:
            print(marker, flush=True)
        return 0
    except DriverError as error:
        print(f"n06a_paired_serving_driver: error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("n06a_paired_serving_driver: interrupted", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
