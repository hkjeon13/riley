"""Select a GPU and one startup policy for the previous diagnostic runner."""

import json
from pathlib import Path
import signal
import sys
import time
from unittest.mock import patch

import run_probe

GPUS = {
    "6": "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7",
    "7": "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0",
}
POLICIES = {"cold": "disabled", "warm": "step-shapes-v1"}


def interrupted(_number, _frame):
    raise KeyboardInterrupt


def run(root, build, index, mode, condition):
    if mode not in ("off", "filtered"):
        raise SystemExit("expected off or filtered")
    name = f"{mode}-{condition}-gpu{index}"
    original_docker, original_ready = run_probe.docker, run_probe.wait_ready
    launched = None

    def docker(*args):
        nonlocal launched
        if args[0] == "start":
            launched = time.monotonic_ns()
        return original_docker(*args)

    def ready(*args, **kwargs):
        result = original_ready(*args, **kwargs)
        duration = (time.monotonic_ns() - launched) / 1e6
        (root / name / "readiness.json").write_text(json.dumps({"container_start_to_ready_ms": duration}) + "\n")
        return result

    with (patch.object(run_probe, "GPU", GPUS[index]), patch.object(run_probe, "ARMS", (name,)),
          patch.object(run_probe, "SERVE", [*run_probe.SERVE, "--startup-warmup", POLICIES[condition]]),
          patch.object(run_probe, "docker", docker), patch.object(run_probe, "wait_ready", ready)):
        return run_probe.run(root, build, root / "workload_startup.py", name)


if __name__ == "__main__":
    stage, build, index, mode, condition = sys.argv[1:]
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run(Path(stage), Path(build), index, mode, condition))
