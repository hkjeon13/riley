"""Derive timing-only replay inputs from the twelve verified GPU cells."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from analyze_prefill_turns import reconstruct
from compare_profile_arms import digest, read, require
from phase_inputs import ORDER, OFFSETS_MS
from summarize_request_stages import read_phases
from verify_phase_gpu import WAVE, verify


def waves(campaign, previous, expected_hash):
    proven = verify(campaign, previous)
    serialized = json.dumps(proven, indent=2, allow_nan=False).encode() + b"\n"
    require(sha256(serialized).hexdigest() == expected_hash, "the prior GPU comparison did not replay byte for byte")
    out = {}
    for arm in ORDER:
        folder = campaign / f"qwen14-{arm}"
        log = folder / "server.log"
        phases = read_phases(log)
        steps, turns, admitted, lengths = reconstruct(phases)
        arm_receipt = proven["arms"][arm]
        cells = {}
        for cell in OFFSETS_MS:
            path = folder / f"{cell}.json"
            measured = read(path)["measured"]
            wave = {row["request_id"]: row["request_index"] for row in measured if row["request_index"] in WAVE}
            checked = arm_receipt["cells"][cell]
            require(set(wave.values()) == WAVE and len(wave) == len(WAVE)
                    and digest(path) == checked["result_sha256"], "wrong GPU cell")
            first = next(identifier for identifier, index in wave.items() if index == 40)
            prefix = steps[turns[first][0][0]:][:9]
            require(len(prefix) == 9 and all(
                    len(step.chunks) == 1 and step.chunks[0].request in wave
                    and step.chunks[0].decode_rows == step.chunks[0].final == 0
                    and step.chunks[0].batch_rows == 1024 for step in prefix),
                    "insufficient prefill-only prefix")
            sent = {}
            for identifier in wave:
                (stamp, flags), = phases[identifier]["command_send"]
                require(not flags and stamp <= admitted[identifier], "invalid send event")
                sent[identifier] = stamp
            origin = min(sent.values())
            arrivals = sorted((index, sent[identifier] - origin, admitted[identifier] - origin,
                               lengths[identifier]) for identifier, index in wave.items())
            plans = [[wave[chunk.request], chunk.first_token, chunk.next_token]
                     for step in prefix for chunk in step.chunks]
            require([[plan[0]] for plan in plans[-2:]] == checked["fifth_wave"]["prefill_turns_8_and_9"],
                    "prefix differs from the published GPU evidence")
            cells[cell] = {
                "requests_index_send_admit_length": arrivals,
                "step_windows_ns": [[step.start - origin, step.end - origin] for step in prefix],
                "observed_plans": plans, "cell_sha256": checked["result_sha256"],
            }
        out[arm] = {"server_log_sha256": arm_receipt["server_log_sha256"], "cells": cells}
    return {"scope": "Request indices, lengths and relative engine timings in the fifth measured wave.",
            "published_comparison_sha256": expected_hash, "arms": out}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("comparison_sha256")
    args = parser.parse_args()
    print(json.dumps(waves(args.campaign, args.original, args.comparison_sha256), indent=2, allow_nan=False))
