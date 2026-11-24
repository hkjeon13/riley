"""Record final requests and IDs from one model, without using future tokens for proposals."""

import argparse
import json
from pathlib import Path

from agent_fleet_replay import inventory
from agent_workflow import replay, stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    endpoints, _ = inventory(json.loads(args.inventory.read_text()))
    address, key = endpoints[0]
    cases = json.loads(args.registration.read_text())["heldout_cases"]
    records = []
    for seed, history, pairs in cases:
        request = None

        def send(host, req, row, event):
            nonlocal request
            if row["turn"] == 2:
                request = {**req, "return_token_ids": True}
                req = request
            return stream(host, req, row, event, bearer=key)

        result = replay(
            address,
            "m",
            seed,
            pairs,
            0.05,
            f"lookup-{seed}",
            history=history,
            send=send,
        )
        records.append({"workflow": result, "final_request": request})
        args.out.write_text(json.dumps(records) + "\n")


if __name__ == "__main__":
    main()
