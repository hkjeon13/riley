"""Run the existing distribution checker with an attested allocator derivative."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import check_distribution_gpu as checker
import verify_soaks
from verify_allocator_image import LIB, LIB_SHA, START, image_config, inspect

PROFILE = "abort_conf:true,dirty_decay_ms:0,muzzy_decay_ms:0,oversize_threshold:0"
GPU6 = "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7"
GOLDENS = {"baseline": "b32bd641dab4e9516a8d3b45de2d77fbde91ad64acad893153f4b6c2090b72a6",
           "holdout": "12d508ed51f03509432e6302b8d10855f56125f4f6b7f31a559cff03eb10b611"}
HOLDOUT = "220f13a7306518f996480c608733623475471ff3721ea3aeb895a2d5f91abd11"
SHELL = r"""
test "$(id -u):$(id -g)" = 65532:65532
test -z "${LD_PRELOAD:-}${MALLOC_CONF:-}${GLIBC_TUNABLES:-}"
test "$(readlink /proc/1/exe)" = /opt/riley/bin/riley
tr '\0' '\n' </proc/1/environ | grep -E '^(LD_PRELOAD|MALLOC_CONF|GLIBC_TUNABLES)='
sha256sum "$1"
awk -v path="$1" '$NF == path { count++ } END { print count+0 }' /proc/1/maps
"""
read = verify_soaks.read
digest = verify_soaks.digest


def bind(root, parent_file):
    parent = read(parent_file)
    package = read(root / "verified-image.json")
    assert package["status"] == parent["status"] == "passed"
    assert package["parent_image_id"] == parent["image_id"] == verify_soaks.IMAGE
    assert package["parent_cpu_report_sha256"] == digest(parent_file)
    assert all(parent[key] == package[key] for key in ("source_revision", "source_archive_sha256", "binaries"))
    assert package["added_files_sha256"][LIB] == LIB_SHA
    return {**parent, "image_id": package["image_id"]}


def probe(container, expected):
    info = checker.inspect_container(container)
    pid = info["State"]["Pid"]
    assert info["Image"] == expected and info["Path"] == START and info["State"]["Running"] and pid > 0
    assert {item["Destination"] for item in info["Mounts"]} == {"/models"}
    observed = checker.docker("exec", container, "/bin/sh", "-ec", SHELL, "proof", LIB).splitlines()
    assert len(observed) == 4 and set(observed[:2]) == {f"LD_PRELOAD={LIB}", f"MALLOC_CONF={PROFILE}"}
    assert observed[2].split() == [LIB_SHA, LIB] and int(observed[3]) > 0
    return {"container_id": container, "host_pid": pid, "image_id": expected,
            "library_sha256": LIB_SHA, "malloc_conf": PROFILE}


def accuracy_inputs(base):
    source = base / "src/benchmarks/evidence"
    baseline = source / "20261119-w4-long/riley/capture.json.gz"
    workload = source / "20261119-w4-holdout/holdout.json.gz"
    golden = source / "20261119-w4-holdout/riley/capture.json.gz"
    assert (digest(baseline), digest(workload), digest(golden)) == (GOLDENS["baseline"], HOLDOUT, GOLDENS["holdout"])
    return baseline, workload, golden


def run(root, parent_file, base, scope):
    spec = bind(root, parent_file)
    adapter = root / "gpu-checker-input.json"
    assert read(adapter) == spec
    original_config, original_ready, original_docker = checker.check_config, checker.wait_ready, checker.docker

    def config(documents, image, revision, archive):
        variant, = documents
        assert variant["Id"] == image == spec["image_id"]
        parent = inspect(verify_soaks.IMAGE)
        original_config([parent], verify_soaks.IMAGE, revision, archive)
        image_config(parent, variant)

    def ready(container, port):
        result = original_ready(container, port)
        (root / f"{scope}-allocator-before.json").write_text(json.dumps(probe(container, spec["image_id"])) + "\n")
        return result

    def docker(*args):
        if args[0] == "stop":
            proof = probe(args[-1], spec["image_id"])
            assert proof == read(root / f"{scope}-allocator-before.json")
            (root / f"{scope}-allocator-after.json").write_text(json.dumps(proof) + "\n")
        return original_docker(*args)

    if scope == "accuracy":
        baseline, workload, golden = accuracy_inputs(base)
        options = {"accuracy_baseline": baseline, "accuracy_holdout": (workload, golden)}
        uuid = GPU6
    else:
        options = {"mixed_soak": scope}
        uuid = verify_soaks.GPUS[scope][1]
    with (patch.object(checker, "check_config", config), patch.object(checker, "wait_ready", ready),
          patch.object(checker, "docker", docker), patch.object(checker, "ACCURACY_CHECK", root / "long_accuracy.py")):
        return checker.check_gpu(adapter, uuid, base / "model", verify_soaks.MODEL, root / scope, True, **options)


def compare_proof(root, scope, image):
    proof = read(root / f"{scope}-allocator-before.json")
    assert proof == read(root / f"{scope}-allocator-after.json") and proof["image_id"] == image
    assert proof["library_sha256"] == LIB_SHA and proof["malloc_conf"] == PROFILE
    return proof


def verify(root, parent_file):
    spec = bind(root, parent_file)
    adapter = root / "gpu-checker-input.json"
    assert read(adapter) == spec
    with (patch.object(verify_soaks, "IMAGE", spec["image_id"]),
          patch.object(verify_soaks, "CPU_REPORT", digest(adapter))):
        report = verify_soaks.verify(root)
    report.update(parent_image_id=verify_soaks.IMAGE, allocator_cpu_sha256=digest(root / "verified-image.json"),
                  allocator_library_sha256=LIB_SHA, allocator_settings=PROFILE)
    for mode, result in report["modes"].items():
        proof = compare_proof(root, mode, spec["image_id"])
        rss_source = read(root / mode / "soak.json")["rss_source"]
        assert proof["container_id"] == rss_source["container_id"] and proof["host_pid"] == rss_source["host_pid"]
        result["allocator_process_proof_sha256"] = digest(root / f"{mode}-allocator-before.json")
    return report


def verify_accuracy(root, parent_file):
    spec = bind(root, parent_file)
    folder = root / "accuracy"
    report = read(folder / "report.json")
    assert report["status"] == "passed" and report["api_checks"] == 40 and report["shutdown_exit"] == 0
    assert report["checkpoint_sha256"] == verify_soaks.MODEL and report["gpu_uuid"] == GPU6
    assert report["riley_binary_sha256"] == spec["binaries"]["/opt/riley/bin/riley"]
    assert report["source_revision"] == spec["source_revision"] and report["image_id"] == spec["image_id"]
    assert report["cpu_report_sha256"] == digest(root / "gpu-checker-input.json")
    assert all(digest(folder / path) == value for path, value in report["artifacts_sha256"].items())
    for label, count in (("baseline", 144), ("holdout", 88)):
        suite = report["accuracy_suites"][label]
        assert suite["verified_requests"] == count and suite["golden_sha256"] == GOLDENS[label]
        assert suite["workload_sha256"] == (HOLDOUT if label == "holdout" else None)
    compare_proof(root, "accuracy", spec["image_id"])
    shutdown = read(folder / "shutdown.json")
    assert not any(shutdown[key] for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
    assert shutdown["allocation"] and not any(shutdown["allocation"].values())
    return {"status": "passed", "source_revision": spec["source_revision"], "image_id": spec["image_id"],
            "riley_binary_sha256": report["riley_binary_sha256"], "accuracy_suites": report["accuracy_suites"],
            "report_sha256": digest(folder / "report.json")}


action, directory, cpu, *args = sys.argv[1:]
root, cpu = Path(directory), Path(cpu)
if action == "run":
    raise SystemExit(run(root, cpu, Path(args[0]), args[1]))
commands = {"bind": bind, "verify": verify, "verify-accuracy": verify_accuracy}
print(json.dumps(commands[action](root, cpu), indent=2, allow_nan=False))
