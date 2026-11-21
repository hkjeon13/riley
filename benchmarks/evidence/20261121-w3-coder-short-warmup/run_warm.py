"""Select one authorized device for the previous hash-pinned diagnostic runner."""

from pathlib import Path
import signal
import sys

import run_probe

GPUS = {
    "6": "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7",
    "7": "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0",
}


def interrupted(_number, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    stage, build, index, mode = sys.argv[1:]
    if mode not in ("off", "filtered"):
        raise SystemExit("expected off or filtered")
    name = f"{mode}-gpu{index}"
    run_probe.GPU = GPUS[index]
    run_probe.ARMS = (name,)
    root = Path(stage)
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run_probe.run(root, Path(build), root / "workload_warm.py", name))
