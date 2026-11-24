"""Replay a deterministic hot, branched and evicted prefix on each GPU independently."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from agent_workflow import stream


def requests():
    template = [
        ("prime", "A", ""),
        ("repeat", "A", ""),
        ("extend", "A", " More details were collected." * 35),
        ("branch", "A", " Separate measurements were collected." * 35),
        *((f"churn-{tag}", tag, "") for tag in "BCDEFGH"),
        ("revisit", "A", ""),
        ("warm-revisit", "A", ""),
    ]
    for label, tag, suffix in template:
        prompt = tag + " Routine latency measurements stayed stable." * 850 + suffix
        yield (
            label,
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": prompt + "\nSummarize in one sentence."}
                ],
                "max_tokens": 16,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_token_ids": True,
            },
        )


def run(replica):
    results = []
    for label, request in requests():
        row = {"case": label}
        row["answer"], calls = stream(
            replica["url"], request, row, bearer=os.environ[replica["bearer_env"]]
        )
        assert not calls
        results.append(row)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        runs = list(
            executor.map(run, json.loads(args.inventory.read_text())["replicas"])
        )
    args.out.write_text(json.dumps(runs) + "\n")
