"""Run the existing GPU qualification against the separately attested allocator image."""

from argparse import ArgumentParser
import json
from pathlib import Path
from unittest.mock import patch

import check_distribution_gpu as checker
from qualify_allocator import check_reservations, golden_inputs
from verify_allocator_image import LIB, LIB_SHA, START, image_config, inspect
from verify_a100_candidate import CHECKPOINT, GPU, ROOTS, RILEY, digest, gpu_evidence, read, require
from verify_a100_candidate import validate_accuracy, validate_soak
from verify_zerodecay import MODES


PROFILE = "abort_conf:true,dirty_decay_ms:0,muzzy_decay_ms:0,oversize_threshold:0"
SCOPES = (*MODES, "accuracy")
SHELL = r"""
test "$(id -u):$(id -g)" = 65532:65532
test -z "${LD_PRELOAD:-}${MALLOC_CONF:-}${GLIBC_TUNABLES:-}"
test "$(readlink /proc/1/exe)" = /opt/riley/bin/riley
tr '\0' '\n' </proc/1/environ | grep -E '^(LD_PRELOAD|MALLOC_CONF|GLIBC_TUNABLES)='
sha256sum "$1"
awk -v path="$1" '$NF == path { count++ } END { print count+0 }' /proc/1/maps
"""


def binding(package, evidence):
    stock_file = evidence / ROOTS["base"] / "cpu-evidence/report.json"
    stock = read(stock_file)
    report_file = package / "verified-image.json"
    report = read(report_file)
    require(report["schema_version"] == "riley.experimental-allocator-image.v1"
            and report["status"] == "passed" and report["parent_image_id"] == stock["image_id"]
            and report["parent_cpu_report_sha256"] == digest(stock_file)
            and report["source_revision"] == stock["source_revision"]
            and report["source_archive_sha256"] == stock["source_archive_sha256"]
            and report["binaries"] == stock["binaries"]
            and report["added_files_sha256"][LIB] == LIB_SHA, "packaged CPU attestation mismatches the parent")
    return {
        # The common GPU checker accepts this input schema. The original CPU
        # attestation and its image ID are recorded separately in the binding.
        "schema_version": stock["schema_version"], "status": report["status"],
        "image_id": report["image_id"], "source_revision": report["source_revision"],
        "source_archive_sha256": report["source_archive_sha256"], "binaries": report["binaries"],
        "experimental_packaged_cpu_attestation_sha256": digest(report_file),
        "parent_image_id": stock["image_id"],
    }


def process_proof(container, expected):
    info = checker.inspect_container(container)
    pid = info["State"]["Pid"]
    require(info["Image"] == expected and info["Path"] == START and info["State"]["Running"]
            and type(pid) is int and pid > 0, "packaged allocator test changed process or image")
    require({entry["Destination"] for entry in info["Mounts"]} == {"/models"},
            "expected only the checkpoint to be mounted")
    observed = checker.docker("exec", container, "/bin/sh", "-ec", SHELL, "proof", LIB).splitlines()
    require(len(observed) == 4 and set(observed[:2]) == {f"LD_PRELOAD={LIB}", f"MALLOC_CONF={PROFILE}"}
            and observed[2].split() == [LIB_SHA, LIB] and int(observed[3]) > 0,
            "the live Riley process did not use the packaged allocator")
    return {"container_id": container, "host_pid": pid, "image_id": expected,
            "library_sha256": LIB_SHA, "malloc_conf": PROFILE, "mounts": ["/models"]}


def run(package, evidence, stage, scope):
    spec = binding(package, evidence)
    base = evidence / ROOTS["base"]
    directory = stage / scope
    checker_input = stage / "gpu-checker-input.json"
    require(read(checker_input) == spec, "the verified CPU report changed since GPU admission")
    orig_config, orig_ready, orig_docker = checker.check_config, checker.wait_ready, checker.docker

    def config(inspected, image, revision, archive):
        require(len(inspected) == 1 and inspected[0]["Id"] == image == spec["image_id"],
                "GPU checker inspected a different packaged image")
        parent = inspect(spec["parent_image_id"])
        orig_config([parent], spec["parent_image_id"], revision, archive)
        image_config(parent, inspected[0])

    def ready(container, port):
        reply = orig_ready(container, port)
        with (directory / "allocator-before.json").open("x") as out:
            json.dump(process_proof(container, spec["image_id"]), out, indent=2)
        return reply

    def docker(*args):
        if args[0] == "stop":
            proof = process_proof(args[-1], spec["image_id"])
            require(proof == read(directory / "allocator-before.json"), "packaged allocator changed during requests")
            with (directory / "allocator-after.json").open("x") as out:
                json.dump(proof, out, indent=2)
        return orig_docker(*args)

    if scope == "accuracy":
        baseline, workload, golden = golden_inputs(evidence)
        options = {"accuracy_baseline": baseline, "accuracy_holdout": (workload, golden)}
    else:
        options = {"mixed_soak": scope}
    accuracy_tool = evidence / ROOTS["accuracy"] / "harness/benchmarks/live/long_accuracy.py"
    with (patch.object(checker, "check_config", config), patch.object(checker, "wait_ready", ready),
          patch.object(checker, "docker", docker), patch.object(checker, "ACCURACY_CHECK", accuracy_tool)):
        return checker.check_gpu(checker_input, GPU, base / "model", CHECKPOINT,
                                 directory / "gpu-evidence", True, **options)


def verify(package, evidence, stage, scopes):
    spec = binding(package, evidence)
    adapter = stage / "gpu-checker-input.json"
    require(read(adapter) == spec, "packaged CPU attestation or GPU input changed")
    expected = {key: spec[key] for key in ("image_id", "source_revision", "source_archive_sha256")}
    expected.update({"riley_binary_sha256": spec["binaries"][RILEY], "cpu_report_sha256": digest(adapter),
                     "checkpoint_sha256": CHECKPOINT, "gpu_uuid": GPU})
    results = {}
    for scope in scopes:
        directory = stage / scope
        receipt = gpu_evidence(directory, expected, int((directory / "exit").read_text()))
        require(receipt["status"] in ("passed", "failed"), f"{scope} GPU report is incomplete")
        proof = read(directory / "allocator-before.json")
        require(proof == read(directory / "allocator-after.json") and proof["image_id"] == expected["image_id"]
                and proof["library_sha256"] == LIB_SHA and proof["malloc_conf"] == PROFILE
                and proof["mounts"] == ["/models"], "the actual allocator process proof differs")
        holders = check_reservations(directory, "after")
        require(holders == check_reservations(directory, "before"), "GPU reservations did not recover")
        if scope == "accuracy":
            validate_accuracy(directory, receipt)
            checks = {"accuracy_suites": receipt["accuracy_suites"]}
        else:
            validate_soak(directory, receipt, scope)
            source = receipt["verified_soak"]["rss_source"]
            require(proof["container_id"] == source["container_id"] and proof["host_pid"] == source["host_pid"],
                    "RSS was checked on a different process")
            checks = {"mixed_soak": receipt["mixed_soak"], "verified_soak": receipt["verified_soak"]}
        results[scope] = {
            **checks, "status": receipt["status"], "api_checks": receipt["api_checks"], "same_reservations": holders,
            "process_allocator": proof, "report_sha256": receipt["gpu_receipt_sha256"],
            "shutdown_metrics": receipt["shutdown_metrics"], "artifacts_sha256": receipt["artifacts_sha256"],
        }
    return {"scope": "GPU qualification of the actual packaged allocator image.",
            "image_binding": spec, "gpu_checker_input_sha256": digest(adapter), "results": results}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bind", "admission", "run", "verify"))
    parser.add_argument("package", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("stage", type=Path)
    parser.add_argument("scope", nargs="*")
    args = parser.parse_args()
    if any(scope not in SCOPES for scope in args.scope):
        parser.error("scope must be cache-on, cache-off or accuracy")
    if args.action in ("admission", "run") and len(args.scope) != 1:
        parser.error("admission and run require exactly one scope")
    if args.action == "bind":
        result = binding(args.package, args.evidence)
    elif args.action == "admission":
        result = check_reservations(args.stage / args.scope[0], "before")
    elif args.action == "run":
        raise SystemExit(run(args.package, args.evidence, args.stage, args.scope[0]))
    else:
        result = verify(args.package, args.evidence, args.stage, args.scope)
    print(json.dumps(result, indent=2, allow_nan=False))
