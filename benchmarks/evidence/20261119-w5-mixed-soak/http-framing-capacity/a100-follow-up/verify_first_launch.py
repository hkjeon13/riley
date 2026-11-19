"""Verify the isolated first framing container at the start of its W5 soak."""

from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True, timeout=20)


def read(path):
    return json.loads(path.read_bytes())


def verify(stage, holder_dir):
    trial = stage / "cache-on/gpu-evidence"
    image = read(stage / "cpu-comparison.json")["image"]
    candidate = image["candidate_image_id"]
    matches = command("docker", "ps", "--no-trunc", "--filter", f"ancestor={candidate}",
                      "--filter", "label=io.riley.gpu-check", "--format", "{{.ID}}").splitlines()
    container, = matches
    inspected, = json.loads(command("docker", "container", "inspect", container))
    args = inspected["Args"]
    if inspected["Image"] != candidate or inspected["State"]["Running"] is not True:
        raise ValueError("the running container differs from the verified image")
    if args[args.index("--prefix-cache-blocks") + 1] != "128":
        raise ValueError("cache-on did not request its pinned block budget")
    allocator_keys = {"LD_PRELOAD", "GLIBC_TUNABLES", "MALLOC_ARENA_MAX"}
    if any(value.split("=", 1)[0] in allocator_keys for value in inspected["Config"]["Env"]):
        raise ValueError("allocator settings differ from the unmodified image workload")
    holder6, holder7 = (json.loads(command("/usr/bin/python3", "-B", str(holder_dir / "control.py"),
                                          "--lease-dir", str(holder_dir / "leases"),
                                          "--index", str(index), "status")) for index in (6, 7))
    pid = inspected["State"]["Pid"]
    if (holder6["pid"] != 2431228 or holder6["phase"] != "active" or holder6["allocation_bytes"] != 64 * 1024**2
            or {process["pid"] for process in holder6["other_processes"]} != {pid}
            or holder7["pid"] != 2431237 or holder7["phase"] != "idle"
            or holder7["allocation_bytes"] != 20 * 1024**3 or holder7["other_processes"]):
        raise ValueError("GPU reservations or workload process differ")
    readiness = read(trial / "ready.json")
    api = read(trial / "api.json")["results"]
    if (readiness["ready"] is not True or readiness["accepting"] is not True or len(api) != 40
            or len({row["name"] for row in api}) != 40 or any(row["passed"] is not True for row in api)):
        raise ValueError("inference live checks did not complete")
    executable, _ = command("docker", "exec", container, "sha256sum", "/proc/1/exe").split()
    if executable != image["candidate_binary_sha256"]:
        raise ValueError("the running binary differs from the CPU package report")
    return {
        "scope": "First framing container after the API suite, during cache-on mixed load.",
        "candidate_image_id": candidate,
        "candidate_binary_sha256": executable,
        "candidate_cpu_report_sha256": image["candidate_cpu_report_sha256"],
        "api_checks_passed": 40,
        "prefix_cache_blocks": 128,
        "allocator_overrides_present": False,
        "gpu": {"physical_index": 6, "holder_pid": holder6["pid"], "riley_host_pid": pid,
                "other_reservation_pid": holder7["pid"], "other_reservation_bytes": holder7["allocation_bytes"]},
        "gpu_artifacts_sha256": {name: digest(trial / name) for name in ("ready.json", "api.json")},
    }


if __name__ == "__main__":
    print(json.dumps(verify(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
