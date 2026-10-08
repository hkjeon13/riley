from __future__ import annotations

import os
import unittest
from dataclasses import dataclass

from riley_reference.observability import NvmlProcessTreeSampler


@dataclass
class _CpuTimes:
    user: float
    system: float


class _Process:
    def __init__(
        self,
        pid: int,
        created: float,
        *,
        user: float,
        system: float,
        children: tuple["_Process", ...] = (),
    ) -> None:
        self.pid = pid
        self._created = created
        self.user = user
        self.system = system
        self._children = children

    def create_time(self) -> float:
        return self._created

    def cpu_times(self) -> _CpuTimes:
        return _CpuTimes(self.user, self.system)

    def children(self, *, recursive: bool) -> list["_Process"]:
        if not recursive:
            raise AssertionError("sampler must include recursive children")
        return list(self._children)


class _Psutil:
    def __init__(self, root: _Process) -> None:
        self._root = root
        self.requested_pid: int | None = None

    def Process(self, pid: int) -> _Process:  # noqa: N802 - mirrors psutil
        self.requested_pid = pid
        return self._root


@dataclass(frozen=True)
class _MemoryInfo:
    used: int


@dataclass(frozen=True)
class _Utilization:
    gpu: float


class _Nvml:
    def __init__(self) -> None:
        self.initialized = False
        self.memory_samples = iter((100, 250))
        self.utilization_samples = iter((10.0, 30.0))

    def nvmlInit(self) -> None:  # noqa: N802 - mirrors pynvml
        self.initialized = True

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index: int) -> str:  # noqa: N802
        if index != 0:
            raise AssertionError("sampler must observe primary GPU zero")
        return "gpu-0"

    def nvmlDeviceGetMemoryInfo(self, handle: str) -> _MemoryInfo:  # noqa: N802
        if handle != "gpu-0":
            raise AssertionError("unexpected NVML handle")
        return _MemoryInfo(next(self.memory_samples))

    def nvmlDeviceGetUtilizationRates(self, handle: str) -> _Utilization:  # noqa: N802
        if handle != "gpu-0":
            raise AssertionError("unexpected NVML handle")
        return _Utilization(next(self.utilization_samples))


class ObservabilityTests(unittest.TestCase):
    def test_recursive_cpu_and_device_wide_nvml_metrics_match_lane_contract(self) -> None:
        child = _Process(202, 2.0, user=0.3, system=0.2)
        root = _Process(
            101,
            1.0,
            user=0.2,
            system=0.1,
            children=(child,),
        )
        psutil = _Psutil(root)
        nvml = _Nvml()
        sampler = NvmlProcessTreeSampler(
            nvml_module=nvml,
            psutil_module=psutil,
            sample_interval_seconds=60.0,
        )

        sampler.start()
        root.user += 0.08
        root.system += 0.02
        child.user += 0.15
        child.system += 0.05
        measurement = sampler.stop(wall_seconds=0.1)

        self.assertTrue(nvml.initialized)
        self.assertEqual(psutil.requested_pid, os.getpid())
        self.assertAlmostEqual(measurement.cpu_utilization_percent, 300.0)
        self.assertEqual(measurement.gpu_utilization_percent, 20.0)
        self.assertEqual(measurement.peak_gpu_memory_bytes, 250)


if __name__ == "__main__":
    unittest.main()
