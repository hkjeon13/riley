"""Run one isolated Riley server until the client releases its lease."""

import json
import subprocess
import sys
import time


def serve(root, holder, pair, i, keydir, generation=None, launcher="run-replica.sh"):
    label = generation or f"gpu{i}"
    receipt = generation or str(i)
    out = pair / label
    ctl = [
        sys.executable,
        str(holder / "control.py"),
        "--lease-dir",
        str(holder / "leases"),
        "--index",
        str(i),
    ]
    with (pair / f"controller-{receipt}.log").open("wb") as log:
        start = time.monotonic()
        child = subprocess.run(
            [
                *ctl,
                "run",
                "--",
                "bash",
                str(root / launcher),
                str(out),
                str(pair),
                str(keydir / str(i)),
            ],
            stdout=log,
            check=False,
            stderr=subprocess.STDOUT,
        )
        (pair / f"allocation-{receipt}.json").write_text(
            json.dumps(
                {
                    "elapsed_s": time.monotonic() - start,
                    "returncode": child.returncode,
                }
            )
            + "\n"
        )
        if child.returncode:
            raise RuntimeError(f"replica {i} failed; see controller log")
        (out / "lease-after.json").write_bytes(
            subprocess.check_output([*ctl, "status"])
        )
