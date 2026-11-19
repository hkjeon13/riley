"""Run the baseline API contract and two fixed schedules on one image."""

from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess
import sys

ORIGINS = ("parent-first", "packaged-first")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tools", action="store_true", required=True)
    args = parser.parse_args()
    harness = Path(os.environ["RILEY_W5_PERF_HARNESS"]).resolve(strict=True)
    reference = Path(os.environ["RILEY_W5_REPLAY_REFERENCE"]).resolve(strict=True)
    arm = os.environ["RILEY_W5_REPLAY_ARM"]
    if arm not in ("parent-first", "packaged-first", "parent-second", "packaged-second"):
        raise ValueError("unknown fixed-arrival arm")
    command = [sys.executable, "-B"]
    subprocess.run([*command, str(harness / "api_check.py"), args.address, args.model,
                    "--out", str(args.out), "--tools"], check=True, timeout=600)
    origins = ORIGINS if arm.endswith("-first") else tuple(reversed(ORIGINS))
    for origin in origins:
        raw = reference / origin / "gpu-evidence/rag-c8-seed2.records.json"
        output = args.out.parent
        subprocess.run([*command, str(Path(__file__).with_name("fixed_arrival.py")), "replay", str(raw),
                        "--address", args.address, "--model", args.model,
                        "--out", str(output / f"{origin}.summary.json"),
                        "--records", str(output / f"{origin}.records.json")], check=True, timeout=360)
        print(f"FIXED_ARRIVAL_COMPLETED {origin}", flush=True)


if __name__ == "__main__":
    main()
