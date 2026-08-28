#!/usr/bin/env python3
"""CPU-only tests for pidfd-based C02 lifecycle signal ownership."""

from __future__ import annotations

import io
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import c02_lifecycle_process_guard_v1 as guard


def _proc_stat(pid: int, start_ticks: int) -> bytes:
    fields = ["S", *(["0"] * 18), str(start_ticks)]
    return f"{pid} (riley) {' '.join(fields)}\n".encode("ascii")


class C02LifecycleProcessGuardV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.proc_root = Path(self.temporary.name)
        self.pid = 4242
        (self.proc_root / str(self.pid)).mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_stat(self, ticks: int) -> None:
        (self.proc_root / str(self.pid) / "stat").write_bytes(_proc_stat(self.pid, ticks))

    def test_reads_a_strict_linux_start_tick(self) -> None:
        self._write_stat(987654)
        self.assertEqual(guard.read_start_ticks(self.pid, proc_root=self.proc_root), 987654)

    def test_pidfd_signal_is_delivered_only_after_matching_start_tick(self) -> None:
        self._write_stat(17)
        sentinel = os.open(self.proc_root / str(self.pid) / "stat", os.O_RDONLY)
        with (
            mock.patch.object(os, "pidfd_open", return_value=sentinel, create=True) as pidfd_open,
            mock.patch.object(signal, "pidfd_send_signal", create=True) as pidfd_signal,
        ):
            guard.signal_if_current(
                self.pid,
                17,
                "TERM",
                proc_root=self.proc_root,
            )
        pidfd_open.assert_called_once_with(self.pid, 0)
        pidfd_signal.assert_called_once_with(sentinel, signal.SIGTERM, None, 0)
        with self.assertRaises(OSError):
            os.fstat(sentinel)

    def test_pid_reuse_start_tick_drift_never_sends_a_signal(self) -> None:
        self._write_stat(18)
        sentinel = os.open(self.proc_root / str(self.pid) / "stat", os.O_RDONLY)
        with (
            mock.patch.object(os, "pidfd_open", return_value=sentinel, create=True),
            mock.patch.object(signal, "pidfd_send_signal", create=True) as pidfd_signal,
        ):
            with self.assertRaises(guard.ProcessGoneError):
                guard.signal_if_current(
                    self.pid,
                    17,
                    "KILL",
                    proc_root=self.proc_root,
                )
        pidfd_signal.assert_not_called()
        with self.assertRaises(OSError):
            os.fstat(sentinel)

    def test_zombie_is_treated_as_exited_before_any_signal(self) -> None:
        fields = ["Z", *(["0"] * 18), "17"]
        (self.proc_root / str(self.pid) / "stat").write_bytes(
            f"{self.pid} (riley) {' '.join(fields)}\n".encode("ascii")
        )
        sentinel = os.open(self.proc_root / str(self.pid) / "stat", os.O_RDONLY)
        with (
            mock.patch.object(os, "pidfd_open", return_value=sentinel, create=True),
            mock.patch.object(signal, "pidfd_send_signal", create=True) as pidfd_signal,
        ):
            with self.assertRaises(guard.ProcessGoneError):
                guard.signal_if_current(
                    self.pid,
                    17,
                    "TERM",
                    proc_root=self.proc_root,
                )
        pidfd_signal.assert_not_called()

    def test_cli_reports_gone_for_a_missing_process_without_signalling(self) -> None:
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            exit_code = guard.main(["--pid", str(self.pid), "--read-start-ticks"])
        self.assertEqual(exit_code, 3)
        self.assertIn("target process is absent", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
