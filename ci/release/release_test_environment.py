"""Explicit prerequisites for historical host integration tests.

Portable parsers/contracts continue to run on the validation Python. A skip is
an unavailable host probe, never a release qualification. The dedicated legacy
host lane sets RILEY_REQUIRE_LEGACY_HOST_TESTS=1 so missing prerequisites fail.
"""
from __future__ import annotations

import functools
import hashlib
import os
import stat
import sys
import unittest
from pathlib import Path

LEGACY_PYTHON = Path("/usr/bin/python3.10")
LEGACY_PYTHON_SHA256 = "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"


def require_linux_proc():
    if sys.platform != "linux" or not Path("/proc/self/fd").is_dir():
        message = "requires Linux /proc descriptor semantics; no GPU required"
        if os.environ.get("RILEY_REQUIRE_LINUX_CONTRACT_TESTS") == "1":
            raise AssertionError(message)
        raise unittest.SkipTest(message)


def legacy_host_unavailable_reason():
    if sys.platform != "linux":
        return "historical RC3 host probe requires Linux and its reviewed Python 3.10 (no GPU)"
    try:
        info = LEGACY_PYTHON.lstat()
        valid = (stat.S_ISREG(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022
                 and hashlib.sha256(LEGACY_PYTHON.read_bytes()).hexdigest() == LEGACY_PYTHON_SHA256)
    except OSError:
        valid = False
    return None if valid else "historical RC3 host probe requires the exact reviewed /usr/bin/python3.10"


def requires_legacy_host(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        reason = legacy_host_unavailable_reason()
        if reason:
            if os.environ.get("RILEY_REQUIRE_LEGACY_HOST_TESTS") == "1":
                raise AssertionError(reason)
            raise unittest.SkipTest(reason)
        return function(*args, **kwargs)
    return run
