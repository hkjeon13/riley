"""Upper-bound exact reusable blocks in every protected Coder-14B prompt."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "live"))
from load import prompt


def tally(cell, binary, tokenizer):
    spec = cell["cell"]
    rng = random.Random(spec["seed"])
    bodies = [
        {
            "model": "m",
            "temperature": 0,
            "max_tokens": spec["max_tokens"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {
                    "role": "user",
                    "content": prompt(rng, rng.choice(spec["prompt_words"])),
                }
            ],
        }
        for _ in range(spec["warmup"] + spec["requests"])
    ]
    for row in cell["schedule"]:
        if (
            hashlib.sha256(
                json.dumps(bodies[row["request_index"]]).encode()
            ).hexdigest()
            != row["input_sha256"]
        ):
            raise ValueError("reconstructed prompt differs")
    out = subprocess.run(
        [binary, tokenizer],
        input="".join(json.dumps(req) + "\n" for req in bodies),
        text=True,
        capture_output=True,
        check=True,
    )
    ids = [json.loads(line)["ids"] for line in out.stdout.splitlines()]
    if len(ids) != len(bodies):
        raise ValueError("missing native encodings")
    # A prefix can match only its lexicographic predecessor or successor most deeply.
    sorted_ids = sorted(range(len(ids)), key=ids.__getitem__)
    credits = {}
    for pos, idx in enumerate(sorted_ids):
        shared = 0
        for neighbor in (
            sorted_ids[max(0, pos - 1) : pos] + sorted_ids[pos + 1 : pos + 2]
        ):
            length = 0
            for left, right in zip(ids[idx], ids[neighbor]):
                if left != right:
                    break
                length += 1
            shared = max(shared, length)
        credits[idx] = min(len(ids[idx]) - 1, shared) // 16 * 16
    sample = []
    for row in cell["schedule"]:
        idx = row["request_index"]
        if len(ids[idx]) != row["prompt_tokens"]:
            raise ValueError("native token count differs from server usage")
        sample.append(
            {"index": idx, "tokens": len(ids[idx]), "oracle_cached_upper": credits[idx]}
        )
    total = sum(r["tokens"] for r in sample)
    upper = sum(r["oracle_cached_upper"] for r in sample)
    return {
        "measured": len(sample),
        "total_prompt_tokens": total,
        "oracle_reusable_tokens": upper,
        "oracle_reusable_fraction": upper / total,
        "max_tokens_reused_in_one_request": max(
            r["oracle_cached_upper"] for r in sample
        ),
        "per_request": sample,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("binary", type=Path)
    p.add_argument("tokenizer", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            {
                k: tally(v, a.binary, a.tokenizer)
                for k, v in json.loads(a.plan.read_text())["cells"].items()
            },
            indent=2,
        )
    )
