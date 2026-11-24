"""Preserve normal streaming timing, then separately request rendered token fingerprints."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from agent_workflow import fingerprint, replay, stream


def trial(address, directory, frozen):
    subprocess.run(
        [
            sys.executable,
            "-B",
            Path(__file__).with_name("fixed_agent_decode.py"),
            address,
            frozen,
            directory / "fixed.json",
        ],
        check=True,
    )
    records = []
    finals = {}
    for seed in (761, 762, 833, 861, 902, 931):

        def send(endpoint, request, row, first_visible, seed=seed):
            if row["turn"] == 2:
                finals[seed] = request
            return stream(endpoint, request, row, first_visible)

        short = seed == 761
        records.append(
            replay(
                address,
                "m",
                seed,
                2 if short else 96,
                0.1,
                f"chain-{seed}",
                history=0 if short else 100,
                send=send,
            )
        )
        (directory / "natural.json").write_text(json.dumps(records) + "\n")
    probes = {}
    for seed, request in finals.items():
        row = {}
        stream(address, {**request, "max_tokens": 1, "return_token_ids": True}, row)
        probes[seed] = {
            "original_input_sha256": fingerprint(request),
            "prompt_ids_sha256": row["prompt_ids_sha256"],
            "prompt_tokens": row["usage"]["prompt_tokens"],
        }
    (directory / "identity.json").write_text(json.dumps(probes) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("directory", type=Path)
    parser.add_argument("frozen", type=Path)
    args = parser.parse_args()
    trial(args.address, args.directory, args.frozen)
