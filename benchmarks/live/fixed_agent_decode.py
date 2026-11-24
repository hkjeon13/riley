"""Match submitted final-turn requests, output lengths and arrival offsets across engines."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    stream,
    turn_request,
)


def freeze(workflow, model):
    tools, prompt, docs = fixture(workflow["seed"], workflow["document_pairs"])
    messages = history_messages(workflow["seed"], workflow.get("history_entries", 0))
    messages.append({"role": "user", "content": prompt})
    for idx, turn in enumerate(workflow["turns"][:2]):
        append_tool_result(messages, [turn["call"]], docs, idx)
    request = turn_request(model, messages, tools, None)
    if fingerprint(request) != workflow["turns"][-1]["input_sha256"]:
        raise ValueError("frozen final request differs from original submission")
    request["max_tokens"] = 64
    return request


def measure(address, label, request, epoch, offset, ids=False):
    time.sleep(max(0, epoch + offset - time.monotonic()))
    row = {"name": label, "offset_ms": offset * 1000}
    options = {**request, "return_token_ids": True} if ids else request
    text, calls = stream(address, options, row)
    count = row["usage"]["completion_tokens"]
    if row["finish"] != "length" or count != 64 or len(row["content_ns"]) < 2 or calls:
        raise ValueError(
            "fixed diagnostic must produce 64 tokens and multiple visible chunks"
        )
    row["text_sha256"] = fingerprint(text)
    row["tpot_ms"] = (row["content_ns"][-1] - row["content_ns"][0]) / 1e6 / (count - 1)
    row["schedule_error_ms"] = (row["sent_ns"] / 1e9 - epoch - offset) * 1000
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    results = []
    for history in (0, 200):
        records = json.loads((args.fixtures / f"workflows-{history}.json").read_text())
        reader, writer = records[-2:]
        specs = [
            ("reader", freeze(reader, "m"), 0),
            ("long-tool-result", freeze(writer, "m"), 0.2),
        ]
        # Separate the all-token-IDs diagnostic from normal SSE timing. Warm
        # both exact final prompts before each measured wave (fixed cache state).
        for label, req, _ in specs:
            results.append(
                {
                    "history": history,
                    "phase": "identity",
                    "result": measure(
                        args.address, label, req, time.monotonic(), 0, True
                    ),
                }
            )
        for repeat in range(3):
            epoch = time.monotonic() + 0.05
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                jobs = [
                    pool.submit(measure, args.address, label, req, epoch, offset)
                    for label, req, offset in specs
                ]
                rows = [job.result() for job in jobs]
            if (
                not rows[0]["content_ns"][0]
                < rows[1]["sent_ns"]
                < rows[0]["content_ns"][-1]
            ):
                raise ValueError(
                    "scheduled second prompt did not overlap first response"
                )
            results.append(
                {
                    "history": history,
                    "phase": "normal",
                    "repeat": repeat,
                    "results": rows,
                }
            )
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
