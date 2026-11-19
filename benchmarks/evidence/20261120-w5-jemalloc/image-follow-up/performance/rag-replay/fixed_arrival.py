"""Replay the same measured RAG request writes against each image."""

import argparse
import asyncio
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import random
import time

from load import one, percentile, prompt, summarize


SEED = 2
WARMUP = 8
MEASURED = 64
LENGTHS = [1500, 3000]
MAX_TOKENS = 128


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def bodies(model):
    rng = random.Random(SEED)
    return [{
        "model": model, "temperature": 0, "max_tokens": MAX_TOKENS, "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt(rng, rng.choice(LENGTHS))}],
    } for _ in range(WARMUP + MEASURED)]


def request_digest(body):
    return sha256(json.dumps(body).encode()).hexdigest()


def read_schedule(reference, requests):
    encoded = reference.read_bytes()
    rows = json.loads(encoded)
    measured = list(range(WARMUP, WARMUP + MEASURED))
    require([row["request_index"] for row in rows] == measured, "expected the complete RAG C8 seed-2 capture")
    for index, row in zip(measured, rows, strict=True):
        require(row["input_sha256"] == request_digest(requests[index]), f"request {index} differs from the closed-loop capture")
        require(all(type(row[key]) in (int, float) and isfinite(row[key]) and row[key] >= 0
                    for key in ("start", "sent_at")) and row["start"] <= row["sent_at"],
                f"request {index} has invalid start and write timestamps")
    origin = min(row["sent_at"] for row in rows)
    clock = {row["request_index"]: row["sent_at"] - origin for row in rows}
    return clock, sha256(encoded).hexdigest()


async def replay(address, model, reference):
    host, raw_port = address.split(":")
    port = int(raw_port)
    requests = bodies(model)
    clock, reference_hash = read_schedule(reference, requests)
    warmup = [{} for _ in range(WARMUP)]
    await asyncio.gather(*(one(host, port, requests[i], time.perf_counter(), warmup[i]) for i in range(WARMUP)))
    require(all(row.get("error") is False for row in warmup), "warmup has an error")
    measured = {index: {} for index in clock}
    start = time.perf_counter()
    tasks = []
    for index in sorted(clock, key=clock.get):
        await asyncio.sleep(max(0, start + clock[index] - time.perf_counter()))
        tasks.append(asyncio.create_task(one(host, port, requests[index], time.perf_counter(), measured[index],
                                             capture_hashes=True)))
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    rows = list(measured.values())
    require(all(row.get("error") is False and row["input_sha256"] == request_digest(requests[index])
                for index, row in measured.items()), "measurement has an error or different input")
    actual_origin = min(row["sent_at"] for row in rows)
    offsets_ms = {index: 1000 * (row["sent_at"] - actual_origin - clock[index])
                  for index, row in measured.items()}
    summary = summarize(rows, elapsed, {"seed": SEED, "warmup": WARMUP, "requests": MEASURED,
                                       "prompt_words": LENGTHS, "max_tokens": MAX_TOKENS, "schedule_sha256": reference_hash})
    summary["relative_write_error_abs_p95_ms"] = percentile([abs(value) for value in offsets_ms.values()], .95)
    summary["relative_write_error_abs_max_ms"] = max(abs(value) for value in offsets_ms.values())
    result = [{"request_index": index, "scheduled_write": clock[index], "relative_write_error_ms": offsets_ms[index],
               **{key: row[key] for key in ("input_sha256", "output_sha256", "prompt_tokens", "completion_tokens",
                                            "finish", "ttft", "tpot", "e2e")}}
              for index, row in measured.items()]
    return summary, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "replay"))
    parser.add_argument("reference", type=Path)
    parser.add_argument("--model", default="m")
    parser.add_argument("--address")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--records", type=Path)
    args = parser.parse_args()
    if args.action == "preflight":
        clock, checksum = read_schedule(args.reference, bodies(args.model))
        print(json.dumps({"reference_sha256": checksum, "matched_requests": len(clock),
                          "scheduled_span_seconds": max(clock.values())}, indent=2))
        return
    if not args.address or args.out is None or args.records is None:
        parser.error("replay requires --address, --out and --records")
    summary, rows = asyncio.run(replay(args.address, args.model, args.reference))
    for path, report in ((args.out, summary), (args.records, rows)):
        with path.open("x") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
