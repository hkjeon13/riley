"""Replay both Qwen2.5-14B RAG write schedules with either QKV plan."""

import asyncio
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import random
import time

from load import one, percentile, prompt, summarize


ORDER = ("riley-first", "qkv-first", "qkv-second", "riley-second")
ORIGINS = ORDER[:2]
SEEDS = (1, 2)
BINARY_SHA256 = "f5872f90e59b4cc99a3fb4e626aa1f6e7de090a039f2bd97a86ce501ee8d1037"
OUTPUT_FIELDS = ("request_index", "input_sha256", "output_sha256", "prompt_tokens", "completion_tokens", "finish")


def require(pass_condition, detail):
    if not pass_condition:
        raise ValueError(detail)


def checked_read(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def payloads(cell):
    rng = random.Random(cell["seed"])
    return [{
        "model": "m", "temperature": 0, "max_tokens": cell["max_tokens"], "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt(rng, rng.choice(cell["prompt_words"]))}],
    } for _ in range(cell["warmup"] + cell["requests"])]


def input_hash(body):
    return sha256(json.dumps(body).encode()).hexdigest()


def outputs(rows):
    return [tuple(row[key] for key in OUTPUT_FIELDS) for row in rows]


def baseline(arm, seed, cell):
    name = f"rag-c8-s{seed}"
    run, digest = checked_read(arm / f"{name}.json")
    receipt, _ = checked_read(arm / "receipt.json")
    require(receipt["status"] == "passed" and receipt["results"][name] == digest, "frozen request receipt differs")
    rows = run["measured"]
    require(run["statistics"]["ok"] == cell["requests"] and run["statistics"]["warmup_ok"] == cell["warmup"]
            and run["statistics"]["errors"] == run["statistics"]["rejected_429"] == 0, "incomplete original run")
    require([row["request_index"] for row in rows] == list(range(cell["warmup"], cell["warmup"] + cell["requests"])),
            "original request IDs differ")
    all_payloads = payloads(cell)
    for row in rows:
        require(row["input_sha256"] == input_hash(all_payloads[row["request_index"]]), "frozen RAG payload differs")
        require(isfinite(row["start"]) and isfinite(row["sent_at"]) and row["start"] <= row["sent_at"],
                "invalid original send time")
    first = min(row["sent_at"] for row in rows)
    offsets = {row["request_index"]: row["sent_at"] - first for row in rows}
    return rows, offsets, digest


def cell_specs(source, run_mode):
    picks = ORIGINS if run_mode.endswith("-first") else tuple(reversed(ORIGINS))
    cells = {}
    for seed in SEEDS:
        spec = {"prompt_words": [1500, 3000], "max_tokens": 128, "concurrency": 8,
                "requests": 64, "warmup": 8, "seed": seed}
        expected = None
        for origin in picks:
            arm = source / f"qwen14-{origin}"
            rows, _, receipt = baseline(arm, seed, spec)
            signature = outputs(rows)
            require(expected is None or signature == expected, "original outputs disagree")
            expected = signature
            cells[f"rag-c8-s{seed}-{origin}"] = {
                **spec, "schedule_origin": str(arm), "source_receipt_sha256": receipt,
            }
    return cells


async def load_cell(address, cell):
    host, raw_port = address.split(":")
    port = int(raw_port)
    bodies = payloads(cell)
    expected_rows, offsets, proof = baseline(Path(cell["schedule_origin"]), cell["seed"], cell)
    require(proof == cell["source_receipt_sha256"], "schedule changed after preflight")
    warmup = [{} for _ in range(cell["warmup"])]
    measured = {index: {} for index in offsets}

    async def issue(index, entry):
        await asyncio.wait_for(one(host, port, bodies[index], time.perf_counter(), entry, capture_hashes=True), timeout=90)

    await asyncio.gather(*(issue(index, row) for index, row in enumerate(warmup)))
    require(all(row.get("error") is False for row in warmup), "RAG replay warmup failed")
    epoch = time.perf_counter()
    tasks = []
    for index in sorted(offsets, key=offsets.get):
        await asyncio.sleep(max(0, epoch + offsets[index] - time.perf_counter()))
        tasks.append(asyncio.create_task(issue(index, measured[index])))
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - epoch
    require(all(row.get("error") is False for row in measured.values()), "RAG replay result failed")
    actual_origin = min(row["sent_at"] for row in measured.values())
    keep = OUTPUT_FIELDS[1:] + ("start", "sent_at", "ttft", "tpot", "e2e")
    records = [{"request_index": index, **{key: row[key] for key in keep},
                "scheduled_write": offsets[index],
                "relative_write_error_ms": 1000 * (row["sent_at"] - actual_origin - offsets[index])}
               for index, row in measured.items()]
    require(outputs(records) == outputs(expected_rows), "RAG replay changed request or output")
    summary = summarize(list(measured.values()), elapsed, cell)
    summary["warmup_ok"] = len(warmup)
    summary["write_error_abs_p95_ms"] = percentile([abs(row["relative_write_error_ms"]) for row in records], .95)
    return {"statistics": summary, "measured": records}
