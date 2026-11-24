"""Run the same externally timed selected catalog application and frozen decode on either server."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent_workflow import replay
from coder_catalog_trial import wave
from fixed_agent_decode import freeze, measure


def fixed_trial(address, frozen, out):
    specs = [
        (name, freeze(work, "m"), offset)
        for name, work, offset in zip(("reader", "long"), frozen, (0, 0.2), strict=True)
    ]
    results = []
    for name, req, _ in specs:
        results.append(
            {
                "phase": "identity",
                "row": measure(address, name, req, time.monotonic(), 0, True),
            }
        )
    for index in range(3):
        for name, req, _ in specs:
            measure(address, name, req, time.monotonic(), 0)
        with ThreadPoolExecutor(2) as pool:
            epoch = time.monotonic() + 0.1
            running = [
                pool.submit(measure, address, name, req, epoch, offset)
                for name, req, offset in specs
            ]
            results.append(
                {
                    "phase": "measured",
                    "index": index,
                    "rows": [f.result() for f in running],
                }
            )
    (out / "fixed.json").write_text(json.dumps(results) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("out", type=Path)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    reg = json.loads((here / "coder-catalog-comparison.json").read_text())
    warm = replay(a.address, "m", 5091, 2, 0, "warm")
    (a.out / "warm.json").write_text(json.dumps(warm) + "\n")
    rows = []
    for size, base in reg["waves"]:
        rows.append(
            wave(
                a.address,
                a.out,
                "selected",
                size,
                base,
                sample=lambda address, path: {"clock_ns": time.monotonic_ns()},
            )
        )
        (a.out / "waves.json").write_text(json.dumps(rows) + "\n")
    frozen = json.loads((here / "frozen.json").read_text())
    try:
        fixed_trial(a.address, frozen, a.out)
    except (OSError, ValueError, KeyError) as exc:
        (a.out / "fixed-error.json").write_text(json.dumps(str(exc)) + "\n")
