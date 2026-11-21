"""Bind the current checker to an isolated stock or allocator image."""
from argparse import ArgumentParser
import json
from pathlib import Path
import signal
from unittest.mock import patch

import check_distribution_gpu as checker
from current_binding import START, binding, image_config, inspect, process_proof
from verify_a100_candidate import CHECKPOINT, GPU, ROOTS
from verify_a100_candidate import read, require
from run_allocator_performance import PARENT_ENTRY, PARENT_PROOF


def run(stage, name):
    evidence = stage.parent
    package = evidence / "w5-current-allocator-20261121.ZVgY3v8o"
    stock = evidence / ROOTS["stock"]
    base = evidence / ROOTS["base"]
    spec = binding(package, evidence)
    run_dir = stage / name
    parent = name == "stock"
    cpu = stock / "cpu-evidence/report.json" if parent else stage / "gpu-checker-input.json"
    if not parent:
        cpu.write_text(json.dumps(spec, indent=2) + "\n")
    expected_image = read(cpu)["image_id"]
    original_config, original_docker, original_ready = checker.check_config, checker.docker, checker.wait_ready

    def config(document, image, revision, archive):
        only, = document
        require(only["Id"] == expected_image == image, "wrong smoke image")
        if parent:
            original_config(document, image, revision, archive)
        else:
            parent_config = inspect(spec["parent_image_id"])
            original_config([parent_config], spec["parent_image_id"], revision, archive)
            image_config(parent_config, only)

    def proof(container):
        info = checker.inspect_container(container)
        require(info["Image"] == expected_image and info["State"]["Running"]
                and info["Path"] == (PARENT_ENTRY if parent else START), "smoke process changed")
        if parent:
            checker.docker("exec", container, "/bin/sh", "-ec", PARENT_PROOF)
            allocator = "glibc"
        else:
            allocator = process_proof(container, expected_image)
        return {"image_id": expected_image, "container_id": container, "host_pid": info["State"]["Pid"],
                "args": info["Args"], "allocator": allocator}

    def docker(*args):
        if args[0] == "stop":
            observed = proof(args[-1])
            require(observed == read(run_dir / "process-before.json"), "smoke process changed")
            (run_dir / "process-after.json").write_text(json.dumps(observed, indent=2) + "\n")
        return original_docker(*args)

    def ready(container, port):
        result = original_ready(container, port)
        (run_dir / "process-before.json").write_text(json.dumps(proof(container), indent=2) + "\n")
        return result

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    live = stock / "harness/benchmarks/live"
    signal.signal(signal.SIGTERM, interrupted)
    with (patch.object(checker, "check_config", config), patch.object(checker, "docker", docker),
          patch.object(checker, "wait_ready", ready), patch.object(checker, "API_CHECK", live / "api_check.py"),
          patch.object(checker, "SOAK_CHECK", live / "soak.py")):
        return checker.check_gpu(cpu, GPU, base / "model", CHECKPOINT,
                                 run_dir / "gpu-evidence", True, serving_smoke=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("name", choices=("stock", "jemalloc"))
    args = parser.parse_args()
    raise SystemExit(run(args.stage, args.name))
