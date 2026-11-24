"""Compare two prefix preparations during a dependent tool call with the real final response."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from urllib.request import urlopen

from agent_workflow import fingerprint, replay, stream


def prepare(request, call, mode):
    messages = list(request["messages"])
    if mode == "boundary":
        messages.extend(
            [
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": ""},
            ]
        )
    return {**request, "messages": messages, "tool_choice": "none", "max_tokens": 1}


def counters(address):
    with urlopen(f"http://{address}/metrics", timeout=10) as res:
        metrics = json.load(res)
    assert metrics["active_requests"] == metrics["waiting_requests"] == 0
    assert not metrics["batch_shapes"]["metrics_degraded"]
    return metrics["batch_shapes"]["by_phase"]


def measure(address, mode, seed, history, pairs, reader_seed, finals):
    before = counters(address)
    release, visible = threading.Event(), threading.Event()
    prefix = {}
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        reader = None
        if reader_seed is not None:
            reader = executor.submit(
                replay,
                address,
                "m",
                reader_seed,
                2,
                0,
                f"reader-{reader_seed}",
                visible,
                release,
                0,
            )
        preview = None

        def start(index, request, row):
            nonlocal preview
            if index == 1:
                if reader is not None:
                    release.set()
                    if not visible.wait(45):
                        raise ValueError("reader did not start")
                if mode != "off":
                    pending = prepare(request, row["call"], mode)
                    preview = executor.submit(stream, address, pending, prefix)

        def send(endpoint, request, row, first_visible):
            if row["turn"] == 2:
                finals[seed] = request
            return stream(endpoint, request, row, first_visible)

        target = replay(
            address,
            "m",
            seed,
            pairs,
            0.1,
            f"chain-{seed}",
            history=history,
            send=send,
            on_tool_start=start,
        )
        release.set()
        if preview:
            try:
                preview.result()
            except (OSError, ValueError) as error:
                prefix["error"] = str(error)
        peer = reader.result() if reader is not None else None
    return {
        "target": target,
        "reader": peer,
        "preview": prefix,
        "before": before,
        "after": counters(address),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("folder", type=Path)
    parser.add_argument("frozen", type=Path)
    args = parser.parse_args()
    mode = os.environ["CODER_WAIT_MODE"]
    assert mode in ("off", "history", "boundary")
    base = Path(__file__).resolve().parent
    config = json.loads((base / "coder-wait-registration.json").read_text())
    subprocess.run(
        [
            sys.executable,
            "-B",
            base / "fixed_agent_decode.py",
            args.address,
            args.frozen,
            args.folder / "fixed.json",
        ],
        check=True,
    )
    cases = [(*case, None) for case in config["cases"]]
    cases.extend((seed, 100, 96, reader) for seed, reader in config["held_out_readers"])
    records, finals = [], {}
    for seed, history, pairs, reader_seed in cases:
        records.append(
            measure(args.address, mode, seed, history, pairs, reader_seed, finals)
        )
        (args.folder / "cases.json").write_text(json.dumps(records) + "\n")
    identities = {}
    for seed, request in finals.items():
        row = {}
        stream(
            args.address, {**request, "max_tokens": 1, "return_token_ids": True}, row
        )
        identities[seed] = {
            "input": fingerprint(request),
            "tokens": row["prompt_ids_sha256"],
        }
    (args.folder / "identity.json").write_text(json.dumps(identities) + "\n")


if __name__ == "__main__":
    main()
