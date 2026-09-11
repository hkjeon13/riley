"""Shared-definition benchmark observability without eager third-party imports."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass


class ObservabilityError(RuntimeError):
    """NVML or recursive process-tree observation failed."""


@dataclass(frozen=True)
class ObservabilityMeasurement:
    cpu_utilization_percent: float
    gpu_utilization_percent: float
    peak_gpu_memory_bytes: int


class NvmlProcessTreeSampler:
    """Sample device-wide NVML metrics and CPU time for the full process tree."""

    def __init__(
        self,
        *,
        nvml_module: object,
        psutil_module: object,
        device_index: int = 0,
        sample_interval_seconds: float = 0.01,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ObservabilityError("observability sample interval must be positive")
        self._nvml = nvml_module
        self._psutil = psutil_module
        self._sample_interval_seconds = sample_interval_seconds
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_error: Exception | None = None
        self._memory_samples: list[int] = []
        self._utilization_samples: list[float] = []
        self._cpu_start: dict[tuple[int, float], float] = {}
        self._cpu_latest: dict[tuple[int, float], float] = {}
        try:
            self._nvml.nvmlInit()
            self._device_handle = self._nvml.nvmlDeviceGetHandleByIndex(device_index)
            self._root_process = self._psutil.Process(os.getpid())
        except Exception as error:
            raise ObservabilityError(
                f"cannot initialize NVML/process-tree sampler: {error}"
            ) from error

    @staticmethod
    def _cpu_seconds(process: object) -> tuple[tuple[int, float], float]:
        pid = getattr(process, "pid", None)
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ObservabilityError("psutil returned an invalid process ID")
        created = float(process.create_time())
        times = process.cpu_times()
        seconds = float(times.user) + float(times.system)
        if not math.isfinite(created) or not math.isfinite(seconds) or seconds < 0:
            raise ObservabilityError("psutil returned invalid process CPU times")
        return (pid, created), seconds

    def _cpu_snapshot(self) -> dict[tuple[int, float], float]:
        try:
            processes = [
                self._root_process,
                *self._root_process.children(recursive=True),
            ]
        except Exception as error:
            raise ObservabilityError(
                f"cannot enumerate benchmark process tree: {error}"
            ) from error
        snapshot: dict[tuple[int, float], float] = {}
        for process in processes:
            try:
                identity, seconds = self._cpu_seconds(process)
            except Exception:
                # A worker can exit between enumeration and observation.
                continue
            snapshot[identity] = seconds
        if not snapshot:
            raise ObservabilityError(
                "benchmark process tree has no observable CPU times"
            )
        return snapshot

    def _sample_once(self) -> None:
        try:
            memory = int(
                self._nvml.nvmlDeviceGetMemoryInfo(self._device_handle).used
            )
            utilization = float(
                self._nvml.nvmlDeviceGetUtilizationRates(self._device_handle).gpu
            )
        except Exception as error:
            raise ObservabilityError(
                f"cannot sample primary GPU with NVML: {error}"
            ) from error
        if memory < 0:
            raise ObservabilityError("NVML returned negative used GPU memory")
        if (
            not math.isfinite(utilization)
            or utilization < 0
            or utilization > 100
        ):
            raise ObservabilityError("NVML returned invalid GPU utilization")
        cpu_snapshot = self._cpu_snapshot()
        with self._lock:
            self._memory_samples.append(memory)
            self._utilization_samples.append(utilization)
            for identity, seconds in cpu_snapshot.items():
                self._cpu_latest[identity] = max(
                    seconds, self._cpu_latest.get(identity, seconds)
                )

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self._sample_interval_seconds):
            try:
                self._sample_once()
            except Exception as error:
                with self._lock:
                    self._thread_error = error
                self._stop_event.set()
                return

    def start(self) -> None:
        if self._thread is not None:
            raise ObservabilityError("observability sampler is already active")
        self._stop_event.clear()
        with self._lock:
            self._thread_error = None
            self._memory_samples = []
            self._utilization_samples = []
            self._cpu_start = self._cpu_snapshot()
            self._cpu_latest = dict(self._cpu_start)
        self._sample_once()
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="riley-hf-observability",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, wall_seconds: float) -> ObservabilityMeasurement:
        thread = self._thread
        if thread is None:
            raise ObservabilityError("observability sampler is not active")
        self._stop_event.set()
        thread.join(timeout=max(1.0, self._sample_interval_seconds * 10))
        self._thread = None
        if thread.is_alive():
            raise ObservabilityError("observability sampler thread did not stop")
        self._sample_once()
        with self._lock:
            thread_error = self._thread_error
            memory_samples = tuple(self._memory_samples)
            utilization_samples = tuple(self._utilization_samples)
            cpu_start = dict(self._cpu_start)
            cpu_latest = dict(self._cpu_latest)
        if thread_error is not None:
            raise ObservabilityError(f"observability sampler failed: {thread_error}")
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ObservabilityError(
                "observability interval must be positive and finite"
            )
        cpu_seconds = math.fsum(
            max(0.0, latest - cpu_start.get(identity, 0.0))
            for identity, latest in cpu_latest.items()
        )
        return ObservabilityMeasurement(
            cpu_utilization_percent=100.0 * cpu_seconds / wall_seconds,
            gpu_utilization_percent=(
                math.fsum(utilization_samples) / len(utilization_samples)
            ),
            peak_gpu_memory_bytes=max(memory_samples),
        )
