"""Replay varied tool records and verify that final answers bind values to source IDs."""

import argparse
import json
import random
from pathlib import Path

from agent_workflow import fingerprint, stream, tool, turn_request


def fixture(seed, count):
    rng = random.Random(seed)
    regions = ("north", "south", "west", "east", "central")
    services = ("rail", "water", "power", "road", "port")
    data = [
        {
            "source": f"F{seed}-{i:04d}",
            "region": rng.choice(regions),
            "service": rng.choice(services),
            "approved": rng.randrange(10000, 90000),
        }
        for i in range(count)
    ]
    targets = [data[count // 5], data[count * 4 // 5]]
    keys = [t["source"] for t in targets]
    ask = (
        f"Report the approved amounts for the two source IDs {keys[0]} and {keys[1]}. "
        "Return only a JSON object mapping each requested source ID to its integer amount."
    )
    service = tool("read_register", "Return the complete register.", {})
    messages = [
        {"role": "user", "content": ask},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_register",
                    "type": "function",
                    "function": {"name": "read_register", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_register",
            "content": json.dumps(data, separators=(",", ":")),
        },
    ]
    req = turn_request("m", messages, [service], None)
    req["max_tokens"] = 96
    return req, {t["source"]: t["approved"] for t in targets}


def extract(text):
    value = text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(value)
    except ValueError:
        return None


def run(address, out):
    rows = []
    for seed, count in [(541, 210), (733, 620), (862, 890)]:
        req, expected = fixture(seed, count)
        measurement = {}
        row = {
            "seed": seed,
            "records": count,
            "expected": expected,
            "request_sha256": fingerprint(req),
        }
        try:
            answer, calls = stream(address, req, measurement, timeout_s=180)
            row.update(
                answer=answer,
                valid=not calls
                and extract(answer) == expected
                and bool(measurement.get("done_ns")),
            )
        except (ValueError, OSError) as exc:
            row.update(valid=False, error=str(exc))
        row["measurement"] = measurement
        rows.append(row)
        out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.address, a.out)
