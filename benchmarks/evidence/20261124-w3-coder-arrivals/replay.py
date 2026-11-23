"""Issue the archived Coder-14B cohorts at response-independent write offsets."""
import asyncio
from hashlib import sha256
import json
from math import isfinite
import random
import time

from load import one, percentile, prompt, summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def payloads(spec):
    rng = random.Random(spec["seed"])
    return [{
        "model": "m", "temperature": 0, "max_tokens": spec["max_tokens"], "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt(rng, rng.choice(spec["prompt_words"]))}],
    } for _ in range(spec["warmup"] + spec["requests"])]


def verify_payloads(spec, rows):
    bodies = payloads(spec)
    # The archived cohorts are indexed after the warmup requests. Validate the
    # full ordered set so a repeated request cannot silently replace another.
    require([row["request_index"] for row in rows] == list(range(spec["warmup"], len(bodies))),
            "the RAG cohort has missing or duplicate request indexes")
    for row in rows:
        body = bodies[row["request_index"]]
        require(sha256(json.dumps(body).encode()).hexdigest() == row["input_sha256"]
                and type(row["offset_ns"]) is int and 0 <= row["offset_ns"] <= 180 * 10**9,
                "the request payload or write offset no longer matches the archived record")
    require(min(row["offset_ns"] for row in rows) == 0, "the schedule lacks a write origin")
    return bodies


async def load_cell(address, plan):
    host, port = address.split(":")
    # Keep the workload configuration used by summarize separate from the
    # archived rows that constrain its exact inputs and timing.
    spec = plan["cell"]
    expected = plan["schedule"]
    bodies = verify_payloads(spec, expected)
    warmup = [{} for _ in range(spec["warmup"])]
    measured = {row["request_index"]: {} for row in expected}

    async def issue(index, row):
        await asyncio.wait_for(one(host, int(port), bodies[index], time.perf_counter(), row,
                                   capture_hashes=True), timeout=90)

    await asyncio.gather(*(issue(index, row) for index, row in enumerate(warmup)))
    require(all(row.get("error") is False for row in warmup), "RAG replay warmup did not complete")
    epoch = time.perf_counter()
    pending = []
    for row in sorted(expected, key=lambda item: item["offset_ns"]):
        offset_s = row["offset_ns"] / 1e9
        await asyncio.sleep(max(0, epoch + offset_s - time.perf_counter()))
        pending.append(asyncio.create_task(issue(row["request_index"], measured[row["request_index"]])))
    await asyncio.gather(*pending)
    elapsed = time.perf_counter() - epoch
    require(all(row.get("error") is False and isfinite(row["sent_at"]) for row in measured.values()),
            "a measured RAG request failed")
    actual_origin = min(row["sent_at"] for row in measured.values())
    fields = ("input_sha256", "output_sha256", "prompt_tokens", "completion_tokens", "finish",
              "start", "sent_at", "ttft", "e2e")
    records = []
    length_mismatches = []
    for original in expected:
        index = original["request_index"]
        row = measured[index]
        require(row["input_sha256"] == original["input_sha256"], f"replayed request {index} differs")
        if any(row[key] != original[key] for key in ("prompt_tokens", "completion_tokens", "finish")):
            length_mismatches.append(index)
        error_ms = 1e3 * (row["sent_at"] - actual_origin - original["offset_ns"] / 1e9)
        records.append({"request_index": index, **{key: row[key] for key in fields},
                        "scheduled_write_ns": original["offset_ns"], "write_error_ms": error_ms})
    stats = summarize(list(measured.values()), elapsed, spec)
    stats["warmup_ok"] = len(warmup)
    stats["relative_write_error_abs_p95_ms"] = percentile([abs(row["write_error_ms"]) for row in records], .95)
    return {"statistics": stats, "measured": records, "length_mismatches": length_mismatches}
