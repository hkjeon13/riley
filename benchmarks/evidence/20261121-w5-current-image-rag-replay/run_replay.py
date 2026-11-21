"""Start one matched image and bind its API and benchmark results to the live process."""

from argparse import ArgumentParser
import json
from pathlib import Path
import signal
from unittest.mock import patch

import check_distribution_gpu as checker
from compare_allocator_performance import ORDER, SERVER_ARGS, expected_image
from current_binding import START, inspect, image_config, process_proof
from verify_a100_candidate import CHECKPOINT, GPU, ROOTS, read, require


PARENT_ENTRY = "/opt/riley/bin/riley-docker"
PARENT_PROOF = r"""
test "$(id -u):$(id -g)" = 65532:65532
test "$(readlink /proc/1/exe)" = /opt/riley/bin/riley
test -z "$(tr '\0' '\n' </proc/1/environ | grep -E '^(LD_PRELOAD|MALLOC_CONF|GLIBC_TUNABLES)=' || :)"
! grep -q libjemalloc /proc/1/maps
"""


def run(package, evidence, stage, name):
    directory = stage / name
    cpu, expected = expected_image(package, evidence, stage, name)
    base = evidence / ROOTS["base"]
    original_config, original_docker, original_ready, original_log = (
        checker.check_config, checker.docker, checker.wait_ready, checker.log_command)
    parent = name.startswith("parent-")

    def config(document, image, revision, archive):
        require(len(document) == 1 and document[0]["Id"] == expected["image_id"] == image,
                "wrong performance image")
        if parent:
            original_config(document, image, revision, archive)
        else:
            parent_id = read(cpu)["parent_image_id"]
            stock = inspect(parent_id)
            original_config([stock], parent_id, revision, archive)
            image_config(stock, document[0])

    def runtime_proof(container):
        info = checker.inspect_container(container)
        state = info["State"]
        require(info["Image"] == expected["image_id"] and state["Running"] and state["Pid"] > 0
                and tuple(info["Args"]) == SERVER_ARGS
                and {entry["Destination"] for entry in info["Mounts"]} == {"/models"}
                and info["Path"] == (PARENT_ENTRY if parent else START), "live performance process differs")
        if parent:
            checker.docker("exec", container, "/bin/sh", "-ec", PARENT_PROOF)
            allocator = "glibc"
        else:
            allocator = process_proof(container, expected["image_id"])
        return {"image_id": info["Image"], "container_id": container, "host_pid": state["Pid"],
                "server_arguments": info["Args"], "allocator": allocator}

    def docker(*args):
        if args[0] == "create":
            start = ("serve", "--model", "/models", "--model-id", "m")
            require(args[-len(start):] == start, "common API server arguments changed")
            return original_docker(*args[:-len(start)], *SERVER_ARGS)
        if args[0] == "stop":
            proof = runtime_proof(args[-1])
            require(proof == read(directory / "process-before.json"), "process changed during performance cells")
            (directory / "process-after.json").write_text(json.dumps(proof, indent=2) + "\n")
        return original_docker(*args)

    def ready(container, port):
        status = original_ready(container, port)
        proof = runtime_proof(container)
        (directory / "process-before.json").write_text(json.dumps(proof, indent=2) + "\n")
        return status

    def log(out, file, command, timeout, *, check=True):
        return original_log(out, file, command, 3600 if file == "api.log" else timeout, check=check)

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (patch.object(checker, "check_config", config), patch.object(checker, "docker", docker),
              patch.object(checker, "wait_ready", ready), patch.object(checker, "log_command", log),
              patch.object(checker, "API_CHECK", stage / "replay_client.py")):
            return checker.check_gpu(cpu, GPU, base / "model", CHECKPOINT, directory / "gpu-evidence", True)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("stage", type=Path)
    parser.add_argument("arm", choices=ORDER)
    args = parser.parse_args()
    raise SystemExit(run(args.package, args.evidence, args.stage, args.arm))
