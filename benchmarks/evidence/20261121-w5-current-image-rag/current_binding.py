"""Bind the unchanged measurement cases to the latest packaged allocator trial."""

import verify_a100_candidate as legacy
from verify_a100_candidate import digest, read, require
from verify_allocator_image import LIB, LIB_SHA, START, image_config, inspect

PROFILE = "abort_conf:true,dirty_decay_ms:0,muzzy_decay_ms:0,oversize_threshold:0"
legacy.ROOTS["stock"] = "w5-current-package-20261121.f6SEbjvA"

SHELL = r"""
test "$(id -u):$(id -g)" = 65532:65532
test -z "${LD_PRELOAD:-}${MALLOC_CONF:-}${GLIBC_TUNABLES:-}"
test "$(readlink /proc/1/exe)" = /opt/riley/bin/riley
tr '\0' '\n' </proc/1/environ | grep -E '^(LD_PRELOAD|MALLOC_CONF|GLIBC_TUNABLES)='
sha256sum "$1"
awk -v path="$1" '$NF == path { count++ } END { print count+0 }' /proc/1/maps
"""


def binding(package, evidence):
    file = evidence / legacy.ROOTS["stock"] / "cpu-evidence/report.json"
    parent = read(file)
    attest = read(package / "verified-image.json")
    soaks = read(package / "verified.json")
    accuracy = read(package / "verified-accuracy.json")
    require(parent["status"] == attest["status"] == accuracy["status"] == soaks["release_gate"] == "passed"
            and attest["parent_image_id"] == parent["image_id"]
            and attest["parent_cpu_report_sha256"] == digest(file)
            and all(attest[key] == parent[key] for key in ("source_revision", "source_archive_sha256", "binaries"))
            and attest["image_id"] == soaks["image_id"] == accuracy["image_id"]
            and attest["added_files_sha256"][LIB] == LIB_SHA and soaks["allocator_settings"] == PROFILE
            and {label: item["verified_requests"] for label, item in accuracy["accuracy_suites"].items()}
            == {"baseline": 144, "holdout": 88}, "latest package is not bound to passing soak and accuracy reports")
    return {**parent, "image_id": attest["image_id"], "parent_image_id": parent["image_id"]}


def check_reservations(directory, phase):
    paths = {6: directory / f"idle-{phase}.json", 7: directory / f"gpu7-{phase}.json"}
    result = {}
    for index, path in paths.items():
        row = read(path)
        require(row["phase"] == "idle" and row["physical_index"] == index
                and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30
                and not row["other_processes"], "GPU reservation did not recover")
        result[str(index)] = {"pid": row["pid"], "start_ticks": row["start_ticks"], "gib": 20}
    return result


def reservation(directory):
    before, after = (check_reservations(directory, phase) for phase in ("before", "after"))
    require(before == after, "GPU reservation owner changed")
    return after["6"]


legacy.reservation = reservation


def process_proof(container, expected):
    import check_distribution_gpu as checker

    info = checker.inspect_container(container)
    pid = info["State"]["Pid"]
    require(info["Image"] == expected and info["Path"] == START and info["State"]["Running"] and pid > 0
            and {item["Destination"] for item in info["Mounts"]} == {"/models"}, "running allocator image differs")
    result = checker.docker("exec", container, "/bin/sh", "-ec", SHELL, "proof", LIB).splitlines()
    require(len(result) == 4 and set(result[:2]) == {f"LD_PRELOAD={LIB}", f"MALLOC_CONF={PROFILE}"}
            and result[2].split() == [LIB_SHA, LIB] and int(result[3]) > 0, "allocator library or settings differ")
    return {"container_id": container, "host_pid": pid, "image_id": expected,
            "library_sha256": LIB_SHA, "malloc_conf": PROFILE, "mounts": ["/models"]}
