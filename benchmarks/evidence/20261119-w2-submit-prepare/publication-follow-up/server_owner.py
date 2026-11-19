"""Record the identity of this trial's Popen child before GPU monitoring."""

import json
import os
from pathlib import Path


def start_ticks(proc):
    return int((proc / "stat").read_text().rpartition(")")[2].split()[19])


def record_child(out, child):
    proc = Path("/proc") / str(child.pid)
    if child.poll() is not None or proc.stat().st_uid != os.getuid():
        raise RuntimeError("cannot bind the profiler trial's child at startup")
    claim = {"pid": child.pid, "start_ticks": start_ticks(proc)}
    temporary = out / "server-owner.json.tmp"
    with temporary.open("x") as handle:
        json.dump(claim, handle)
    temporary.replace(out / "server-owner.json")


def is_owned(path, pid, after_ticks):
    try:
        claim = json.loads(path.read_bytes())
    except FileNotFoundError:
        return False
    if claim["pid"] != pid or claim["start_ticks"] < after_ticks:
        return False
    proc = Path("/proc") / str(pid)
    try:
        return (proc.stat().st_uid == os.getuid() and start_ticks(proc) == claim["start_ticks"])
    except FileNotFoundError:
        # A GPU accounting row can outlive the exact child that created it.
        return True
