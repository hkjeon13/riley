"""Capture mapping-level RSS around the packaged soak and an idle glibc trim."""

import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import time

from attribute_anon_rss import mapping_class


HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ([rwxps-]{4}) ([0-9a-f]+) ([0-9a-f]+:[0-9a-f]+) +([0-9]+)\s*(.*?)\s*$")
FIELDS = ("Rss", "Anonymous", "Private_Dirty", "AnonHugePages")
REQUEST = "/tmp/riley-w5-trim.request"
RECEIPT = "/tmp/riley-w5-trim.receipt.json"


def mount_options():
    library = Path(os.environ["RILEY_W5_TRIM_LIBRARY"]).resolve(strict=True)
    if not library.is_file() or any(char in str(library) for char in (",", "\r", "\n")):
        raise ValueError("expected a readable diagnostic library")
    return ("--mount", f"type=bind,source={library},target=/tmp/w5-trim-probe.so,readonly",
            "--env", "LD_PRELOAD=/tmp/w5-trim-probe.so",
            "--env", "RILEY_W5_TRIM_EXE=/opt/riley/bin/riley")


def capture_smaps(open_proc, pid, path):
    with open_proc(pid, "smaps") as stream:
        raw = stream.read().encode()
    with path.open("xb") as output:
        output.write(gzip.compress(raw, mtime=0))
    return read_mappings(raw.decode().splitlines())


def capture_point(open_proc, pid, path):
    if path is not None:
        try:
            capture_smaps(open_proc, pid, path)
        except (OSError, ValueError) as error:
            path.with_suffix(".error.txt").write_text(str(error) + "\n")


def read_mappings(lines):
    mappings = {}
    identity = None
    fields = {}

    def store():
        if identity is None:
            return
        if set(fields) != set(FIELDS) or identity in mappings:
            raise ValueError("missing mapping metrics or duplicate mapping")
        *_, permissions, _offset, _device, _inode, name = identity
        mappings[identity] = (mapping_class(name, permissions), fields)

    for line in lines:
        header = HEADER.fullmatch(line)
        if header:
            store()
            identity = header.groups()
            fields = {}
            continue
        key, _, body = line.partition(":")
        if key in FIELDS:
            value, unit = body.split()
            if identity is None or unit != "kB" or key in fields:
                raise ValueError("invalid mapping metric")
            fields[key] = int(value)
    store()
    if not mappings:
        raise ValueError("no mappings read")
    return mappings


def summarize(snapshots):
    totals = {
        capture: {field: sum(values[field] for _, values in mappings.values()) for field in FIELDS}
        for capture, mappings in snapshots.items()
    }
    mid, drain, before, after = (snapshots[point] for point in ("midpoint", "drain", "pretrim", "posttrim"))
    shared = mid.keys() & drain.keys() & before.keys() & after.keys()
    growing = []
    for identity in shared:
        label = mid[identity][0]
        counts = [point[identity][1]["Anonymous"] for point in (mid, drain, before, after)]
        growth = counts[1] - counts[0]
        if growth > 0:
            growing.append({"class": label, "mapped_kib": (int(identity[1], 16) - int(identity[0], 16)) // 1024,
                            "growth_during_soak_kib": growth,
                            "drain_to_probe_kib": counts[2] - counts[1],
                            "reclaimed_during_probe_kib": counts[2] - counts[3]})
    growing.sort(key=lambda row: (row["growth_during_soak_kib"], row["class"], row["mapped_kib"],
                                  row["drain_to_probe_kib"], row["reclaimed_during_probe_kib"]), reverse=True)
    return {
        "totals_kib": totals,
        "mapping_counts": {point: len(maps) for point, maps in snapshots.items()},
        "mappings_with_same_extent_and_metadata": len(shared),
        "growing_retained_mappings": growing,
        "growing_retained_mappings_reclaimed_kib": sum(max(0, row["reclaimed_during_probe_kib"]) for row in growing),
        "growing_retained_mappings_idle_drift_kib": sum(row["drain_to_probe_kib"] for row in growing),
    }


def run_after_drain(docker, open_proc, container, pid, out):
    soak = json.loads((out / "soak.json").read_bytes())
    if soak["seconds"] != 1800 or not soak["checks"]["idle_after_drain"] or soak["problem_count"] != 0:
        raise ValueError("the trim probe requires a drained 30-minute workload with no unexpected errors")
    if soak["smaps_rollup_second_half"]["rss_growth_kib"] is None:
        raise ValueError("the original strict RSS check lacks complete measurements")
    snapshots = {}
    for point in ("midpoint", "drain"):
        path = out / f"smaps-{point}.txt.gz"
        snapshots[point] = read_mappings(gzip.decompress(path.read_bytes()).decode().splitlines())
    snapshots["pretrim"] = capture_smaps(open_proc, pid, out / "smaps-pretrim.txt.gz")
    docker("exec", container, "/bin/sh", "-ec", f"umask 077; set -C; : > {REQUEST}")
    deadline = time.monotonic() + 15
    while True:
        evidence = docker("exec", container, "/bin/sh", "-c", f"test ! -f {RECEIPT} || cat {RECEIPT}")
        if evidence.strip():
            receipt = json.loads(evidence)
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Riley PID 1 did not return its trim receipt")
        time.sleep(0.25)
    if receipt["pid"] != 1 or receipt["malloc_trim_return"] not in (0, 1):
        raise ValueError("the allocator receipt did not come from Riley PID 1")
    snapshots["posttrim"] = capture_smaps(open_proc, pid, out / "smaps-posttrim.txt.gz")
    result = {"riley_host_pid": pid, "container_id": container, "in_process_trim": receipt,
              "original_rss_gate": soak["checks"]["rss_growth_second_half_at_most_zero"],
              "original_rss_growth_kib": soak["smaps_rollup_second_half"]["rss_growth_kib"],
              **summarize(snapshots)}
    with (out / "allocator-trim.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def checked_probe(docker, container, pid, out):
    from functools import partial
    from soak import open_container_proc

    try:
        probe = run_after_drain(docker, partial(open_container_proc, container), container, pid, out)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return {"artifact": "allocator-trim.json", "original_rss_gate": probe["original_rss_gate"]}
