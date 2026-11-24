"""Measure production tokenization of frozen Coder requests in paired new processes."""

import argparse
import json
import random
import subprocess
from hashlib import sha256
from pathlib import Path
from statistics import median

from fixed_agent_decode import freeze


def digest(ids):
    return sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def cases(records, source, followup):
    # Preserve the registered application's complete final HTTP requests.
    result = {}
    for row in records:
        work = row["work"]
        req = freeze(work, "m")
        req["max_tokens"] = 384
        result["history" if work["history_entries"] else "short"] = req
    rng = random.Random(2607)
    payloads = {
        "tool_json": json.dumps(
            [
                {"id": i, "region": i % 13, "status": "completed", "service": "catalog"}
                for i in range(1024)
            ]
        ),
        "source": source,
        "unicode": "리전별 관측값. cafe\u0301 & Δ latency 🧪.\n" * 650,
        "unique": " ".join(
            "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=14))
            for _ in range(2600)
        ),
    }
    if followup:
        other = random.Random(91731)
        identifiers = [
            "".join(other.choices("abcdefghijklmnopqrstuvwxyz", k=9))
            for _ in range(3100)
        ]
        payloads.update(
            {
                "short_tool": json.dumps(
                    [{"p95_ms": other.randrange(20, 90)} for _ in range(12)]
                ),
                "distinct": " ".join(identifiers),
                "mixed": " ".join(identifiers[:210]) + "\n" + payloads["tool_json"],
            }
        )
    for name, payload in payloads.items():
        result[name] = {
            "model": "m",
            "messages": [{"role": "user", "content": payload}],
        }
    return result


def run(binary, tokenizer, prompts, order):
    sent = "".join(json.dumps(prompts[k], ensure_ascii=False) + "\n" for k in order)
    rows = subprocess.run(
        [binary, tokenizer], input=sent, capture_output=True, text=True, check=True
    )
    return [json.loads(line) for line in rows.stdout.splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "tokenizer", "client", "source", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--followup", action="store_true")
    a = p.parse_args()
    cfg = json.loads(
        Path(__file__)
        .with_name(
            "piece-cache-followup.json"
            if a.followup
            else "piece-cache-registration.json"
        )
        .read_text()
    )
    prompts = cases(json.loads(a.client.read_text()), a.source.read_text(), a.followup)
    times = []
    expected = {}
    for block, modes in enumerate(cfg["orders"]):
        # First pass warms the model's CPU data and is not timed.
        order = list(prompts)
        rng = random.Random(block)
        for _ in range(cfg["repeats"]):
            wave = list(prompts)
            rng.shuffle(wave)
            order.extend(wave)
        for mode in modes:
            rows = run(getattr(a, mode), a.tokenizer, prompts, order)
            durations = {k: [] for k in prompts}
            for index, (name, row) in enumerate(zip(order, rows, strict=True)):
                if name not in expected:
                    expected[name] = row["ids"]
                assert expected[name] == row["ids"], (block, mode, name)
                if index >= len(prompts):
                    durations[name].append(row["encode_ns"])
            times.append({"block": block, "mode": mode, "samples_ns": durations})
    summary = {}
    for name, ids in expected.items():
        ratios = []
        for block in range(len(cfg["orders"])):
            ms = {
                r["mode"]: median(r["samples_ns"][name])
                for r in times
                if r["block"] == block
            }
            ratios.append(ms["candidate"] / ms["baseline"])
        summary[name] = {
            "tokens": len(ids),
            "ids_sha256": digest(ids),
            "median_ratio": median(ratios),
            "paired_ratios": ratios,
        }
    a.out.write_text(
        json.dumps({"summary": summary, "timings": times}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
