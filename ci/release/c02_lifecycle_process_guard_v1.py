#!/usr/bin/env python3
"""Fail-closed PID/start-tick guard for the C02 lifecycle supervisor.

The lifecycle runner launches a host child and must never send a signal merely
because the numeric PID still exists: it can have been reused. This helper
reads the Linux proc stat start tick strictly and, for a signal, opens a pidfd
before that read and delivers through pidfd_send_signal. If the original
process disappears or a PID is reused in either race window, the pidfd
addresses only the original task and no replacement receives the signal.

It has no GPU, network, evidence, or qualification behavior.
"""

from __future__ import annotations

import argparse
import os
import signal
import stat
import sys
from pathlib import Path
from typing import NoReturn, Sequence


MAX_PROC_STAT_BYTES = 64 * 1024
SIGNALS = {"TERM": signal.SIGTERM, "KILL": signal.SIGKILL}


class ProcessGuardError(ValueError):
    """The target process cannot safely be identified or signalled."""


class ProcessGoneError(ProcessGuardError):
    """The original process is gone, so a signal must not be sent."""


def _fail(message: str) -> NoReturn:
    raise ProcessGuardError(message)


def _positive_decimal(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail(f"{label} must be a positive integer")
    return value


def _parse_proc_stat(raw: bytes, expected_pid: int) -> int:
    """Return Linux stat field 22, robust to parentheses in the comm field."""

    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"/proc stat is not ASCII: {error}")
    first_space = text.find(" ")
    closing_parenthesis = text.rfind(")")
    fields = text[closing_parenthesis + 2 :].split() if closing_parenthesis >= 0 else []
    if (
        first_space < 1
        or closing_parenthesis < first_space + 2
        or text[:first_space] != str(expected_pid)
        or len(fields) <= 19
        or not fields[19].isdigit()
    ):
        _fail("/proc stat lacks a valid PID/start-tick tuple")
    if fields[0] in {"Z", "X"}:
        raise ProcessGoneError("target process exited and awaits reaping")
    if len(fields[0]) != 1:
        _fail("/proc stat has an invalid process state")
    start_ticks = int(fields[19])
    if start_ticks < 1:
        _fail("/proc stat start ticks must be positive")
    return start_ticks


def _read_proc_stat_bytes(pid: int, *, proc_root: Path = Path("/proc")) -> bytes:
    """Read one proc stat leaf with no symlink traversal or unbounded input."""

    _positive_decimal(pid, "PID")
    try:
        root = os.fspath(proc_root)
    except TypeError:
        _fail("proc root must be a filesystem path")
    if type(root) is not str or not os.path.isabs(root):
        _fail("proc root must be an absolute path")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if not hasattr(os, "O_NOFOLLOW"):
        _fail("O_NOFOLLOW is unavailable")
    flags |= os.O_NOFOLLOW
    path = os.path.join(root, str(pid), "stat")
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as error:
        raise ProcessGoneError("target process is absent") from error
    except ProcessLookupError as error:
        raise ProcessGoneError("target process is absent") from error
    except OSError as error:
        _fail(f"cannot safely open target /proc stat: {error}")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _fail("target /proc stat is not a regular file")
        chunks: list[bytes] = []
        remaining = MAX_PROC_STAT_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_PROC_STAT_BYTES:
            _fail("target /proc stat exceeds the maximum size")
        if os.read(descriptor, 1):
            _fail("target /proc stat changed while reading")
        return raw
    finally:
        os.close(descriptor)


def read_start_ticks(pid: int, *, proc_root: Path = Path("/proc")) -> int:
    """Read and validate the current target process start tick."""

    checked_pid = _positive_decimal(pid, "PID")
    return _parse_proc_stat(
        _read_proc_stat_bytes(checked_pid, proc_root=proc_root),
        checked_pid,
    )


def signal_if_current(
    pid: int,
    expected_start_ticks: int,
    signal_name: str,
    *,
    proc_root: Path = Path("/proc"),
) -> None:
    """Send TERM/KILL only through a pidfd bound to the expected process."""

    checked_pid = _positive_decimal(pid, "PID")
    checked_ticks = _positive_decimal(expected_start_ticks, "expected start ticks")
    if signal_name not in SIGNALS:
        _fail("signal must be TERM or KILL")
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if not callable(pidfd_open) or not callable(pidfd_send_signal):
        _fail("pidfd signalling is unavailable on this host")
    try:
        pidfd = pidfd_open(checked_pid, 0)
    except ProcessLookupError as error:
        raise ProcessGoneError("target process exited before pidfd open") from error
    except OSError as error:
        _fail(f"cannot open a pidfd for the target process: {error}")
    try:
        observed_ticks = read_start_ticks(checked_pid, proc_root=proc_root)
        if observed_ticks != checked_ticks:
            raise ProcessGoneError(
                "target PID start ticks changed; refusing a reused process ID"
            )
        try:
            pidfd_send_signal(pidfd, SIGNALS[signal_name], None, 0)
        except ProcessLookupError as error:
            raise ProcessGoneError("target process exited before signal delivery") from error
        except OSError as error:
            _fail(f"cannot deliver {signal_name} through the target pidfd: {error}")
    finally:
        os.close(pidfd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--read-start-ticks", action="store_true")
    action.add_argument("--signal", choices=sorted(SIGNALS))
    parser.add_argument("--expected-start-ticks", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.read_start_ticks:
            if args.expected_start_ticks is not None:
                _fail("--expected-start-ticks is only valid with --signal")
            print(read_start_ticks(args.pid))
            return 0
        if args.expected_start_ticks is None:
            _fail("--signal requires --expected-start-ticks")
        signal_if_current(args.pid, args.expected_start_ticks, args.signal)
    except ProcessGoneError as error:
        print(f"C02 lifecycle process guard: {error}", file=sys.stderr)
        return 3
    except (ProcessGuardError, OSError) as error:
        print(f"C02 lifecycle process guard: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
