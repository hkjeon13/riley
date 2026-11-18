"""Recognize a previously bound server during its terminal NVML samples."""

from pathlib import Path


def observed_exit(pid, binding, proc_root=Path("/proc")):
    if binding is None or binding[0] != pid:
        return False
    try:
        stat = (proc_root / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return True
    fields = stat.rpartition(")")[2].split()
    return len(fields) > 19 and fields[0] in ("Z", "X", "x") and fields[19] == binding[1]
