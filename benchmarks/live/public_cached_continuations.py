"""Replay a preceding prefix and its actual continuation at a public workload's token lengths."""

import argparse
import itertools
import json
import subprocess
from pathlib import Path

from agent_workflow import fingerprint, stream
from public_agent_shapes import body


def prepare(source, binary, tokenizer):
    rows = [r for r in source["rows"] if r["set"] == "extended"]

    def encode(requests):
        lines = "".join(json.dumps(req) + "\n" for req in requests)
        p = subprocess.run(
            [binary, tokenizer], input=lines, text=True, capture_output=True, check=True
        )
        return [json.loads(line)["ids"] for line in p.stdout.splitlines()]

    def previews(offsets):
        return [body(r["index"], n, 1) for r, n in zip(rows, offsets, strict=True)]

    full = encode(
        [body(r["index"], r["repetitions"], r["next_completion_tokens"]) for r in rows]
    )
    prefix = encode(previews([1] * len(rows)))
    overheads = [
        sum(1 for _ in itertools.takewhile(lambda pair: pair[0] == pair[1], zip(a, b)))
        - 1
        for a, b in zip(full, prefix, strict=True)
    ]
    offsets = [
        max(0, min(r["repetitions"], r["next_cached_tokens"] - overhead))
        for r, overhead in zip(rows, overheads, strict=True)
    ]
    requests = previews(offsets)
    encoded = encode(requests)
    result = []
    for r, a, b, n, req in zip(rows, full, encoded, offsets, requests, strict=True):
        common = sum(
            1 for _ in itertools.takewhile(lambda pair: pair[0] == pair[1], zip(a, b))
        )
        target = r["next_cached_tokens"]
        if common != target or len(a) != r["encoded_prompt_tokens"]:
            raise ValueError(
                f"token-prefix mismatch for case {r['index']}: {common} vs {target}"
            )
        result.append(
            {
                **r,
                "prefix_repetitions": n,
                "prefix_sha256": fingerprint(req),
                "common_tokens": common,
                "reusable_whole_blocks_tokens": 16 * (common // 16),
            }
        )
    return {"rows": result, "source_sha256": source["source_sha256"]}


def execute(address, manifest, out):
    records = []
    for row in manifest["rows"]:
        pair = {}
        for key, req, expected in [
            (
                "prior",
                body(row["index"], row["prefix_repetitions"], 1),
                row["prefix_sha256"],
            ),
            (
                "continuation",
                body(row["index"], row["repetitions"], row["next_completion_tokens"]),
                row["request_sha256"],
            ),
        ]:
            if fingerprint(req) != expected:
                raise ValueError("immutable request changed")
            m = {}
            try:
                answer, calls = stream(address, req, m)
                m["answer_sha256"] = fingerprint(answer)
                m["complete"] = not calls
            except (OSError, ValueError) as error:
                m["error"] = str(error)
            pair[key] = m
        records.append({"case": row["index"], **pair})
        out.write_text(json.dumps(records) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("source", type=Path)
    p1.add_argument("binary", type=Path)
    p1.add_argument("tokenizer", type=Path)
    p2 = sub.add_parser("run")
    p2.add_argument("address")
    p2.add_argument("manifest", type=Path)
    p2.add_argument("out", type=Path)
    args = p.parse_args()
    if args.command == "prepare":
        print(
            json.dumps(
                prepare(
                    json.loads(args.source.read_text()), args.binary, args.tokenizer
                ),
                indent=2,
            )
        )
    else:
        execute(args.address, json.loads(args.manifest.read_text()), args.out)
