"""Schedule an unchanged reader alongside targets whose tool decisions belong to the application."""

import argparse
import json
import time
from pathlib import Path

from agent_workflow import replay
from coder_catalog_trial import wave


def sample(address, path):
    return {"clock_ns": time.monotonic_ns()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("out", type=Path)
    p.add_argument("mode", choices=["model", "application"])
    a = p.parse_args()
    reg = json.loads(
        Path(__file__).with_name("named-burst-registration.json").read_text()
    )
    warm = replay(a.address, "m", 5091, 2, 0, "warm")
    (a.out / "warm.json").write_text(json.dumps(warm) + "\n")
    waves = []
    for size, base in reg["waves"]:
        waves.append(
            wave(
                a.address,
                a.out,
                "selected",
                size,
                base,
                sample=sample,
                application_targets=a.mode == "application",
            )
        )
        (a.out / "waves.json").write_text(json.dumps(waves) + "\n")
