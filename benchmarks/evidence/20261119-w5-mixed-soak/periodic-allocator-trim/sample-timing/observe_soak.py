"""Record when the original packaged soak reads RSS, then capture adjacent maps."""

import gzip
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import soak


def time_namespace(container):
    host = Path("/proc/self/ns/time").readlink().as_posix()
    remote = subprocess.check_output(
        ["docker", "exec", container, "readlink", "/proc/1/ns/time"], text=True, timeout=10,
    ).strip()
    if host != remote or not re.fullmatch(r"time:\[\d+\]", host):
        raise ValueError("the host and Riley PID 1 do not share a monotonic clock")
    return host


def timed_rollup(read_rollup, out, host_pid, monotonic_ns=time.monotonic_ns):
    points = iter(("midpoint", "drain"))

    def read(pid, *, open_proc):
        point = next(points)
        before = monotonic_ns()
        original = read_rollup(pid, open_proc=open_proc)
        after = monotonic_ns()
        if pid != host_pid:
            raise ValueError("the observer received a different host PID")
        name = f"smaps-maps-{point}.txt.gz"
        maps_start = monotonic_ns()
        try:
            with open_proc(pid, "smaps") as stream:
                maps = stream.read().encode()
            maps_end = monotonic_ns()
            with (out / name).open("xb") as stream:
                stream.write(gzip.compress(maps, mtime=0))
        except (OSError, ValueError) as error:
            print(f"W5-RSS-MAPS-ERROR {point} {type(error).__name__}", flush=True)
            return original
        record = {
            "point": point, "container_pid": 1, "host_pid": pid,
            "rollup_start_ns": before, "rollup_end_ns": after,
            "map_start_ns": maps_start, "map_end_ns": maps_end, "maps": name,
            "rss_kib": original["rss_kib"], "anonymous_kib": original["anonymous_kib"],
        }
        print("W5-RSS-SAMPLE " + json.dumps(record), flush=True)
        return original

    return read


def main():
    args = sys.argv[1:]
    container = args[args.index("--docker-container") + 1]
    out = Path(args[args.index("--out") + 1]).parent
    if not re.fullmatch(r"[0-9a-f]{64}", container):
        raise ValueError("the observer requires the full ID of the checked container")
    print("W5-CLK " + time_namespace(container), flush=True)
    host_pid = int(args[args.index("--pid") + 1])
    soak.read_smaps_rollup = timed_rollup(soak.read_smaps_rollup, out, host_pid)
    return soak.main()


if __name__ == "__main__":
    raise SystemExit(main())
