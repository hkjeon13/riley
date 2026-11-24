"""Test an explicit source citation instruction on unchanged tool data and historic cases."""

import argparse
import json
from pathlib import Path

from agent_workflow import replay, stream

INSTRUCTION = (
    "Cite each report in the explanation using its exact id from the tool result. "
    "Place each id in parentheses immediately after the measurement it supports."
)


def with_citations(request):
    return {
        **request,
        "messages": [
            *request["messages"],
            {"role": "user", "content": INSTRUCTION},
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("out", type=Path)
    p.add_argument("registration", type=Path)
    args = p.parse_args()
    config = json.loads(args.registration.read_text())
    mode = config["order"].get(args.out.name)
    if mode is None:
        return

    def send(address, request, row, first):
        if mode == "explicit" and row["turn"] == 2:
            request = with_citations(request)
        return stream(address, request, row, first)

    results = []
    for item in config["cases"]:
        r = replay(
            args.address,
            "m",
            item["seed"],
            item["pairs"],
            0,
            item["group"],
            history=item["history"],
            send=send,
        )
        results.append(r)
        (args.out / "citations.json").write_text(
            json.dumps({"mode": mode, "records": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
