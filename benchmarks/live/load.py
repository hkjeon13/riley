#!/usr/bin/env python3
"""Open- or closed-loop streaming load for any OpenAI-compatible server.

usage: load.py HOST:PORT MODEL --prompt-words N[,N...] --max-tokens N
               (--concurrency C | --rate R [--burst ON_S,OFF_S])
               --requests N [--warmup N] [--seed S] [--out FILE] [--extra JSON]
               [--shared-prefix-words N] [--records FILE]

Closed loop keeps C requests in flight. Open loop sends Poisson arrivals at R
requests per second whatever the server is doing, so queueing shows up in
latency instead of slowing the client. With --burst the same mean rate arrives
only during ON seconds of each ON+OFF cycle.
"""
import argparse
import asyncio
import json
import random
import time

WORDS = """the river market station garden winter engine letter mountain copper signal harbor forest
planet window silver thread summer lantern village theory circuit pattern shadow bridge ocean memory
compass orchard archive meadow glacier canyon workshop library orbit granite ribbon valley chapter""".split()
TOPICS = ["a lighthouse keeper", "a railway engineer", "a village doctor", "an astronomer", "a ship's cook",
          "a mapmaker", "a beekeeper", "a clock repairer"]


def document(rng, words):
    text = []
    while len(text) < words:
        sentence = [rng.choice(WORDS) for _ in range(rng.randint(6, 14))]
        text.extend(sentence[:-1] + [sentence[-1] + "."])
    return " ".join(text[:words])


def prompt(rng, words):
    topic = rng.choice(TOPICS)
    if words <= 40:
        return f"Write a detailed story about {topic}. {document(rng, words)}"
    return (f"Here is a document:\n\n{document(rng, words)}\n\n"
            f"Using only the document above as inspiration, write a detailed story about {topic}.")


async def one(host, port, body, started_at, record):
    request = json.dumps(body).encode()
    head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(request)}\r\nConnection: close\r\n\r\n").encode()
    times = []
    usage = {}
    finish = None
    try:
        reader, writer = await asyncio.open_connection(host, port, limit=1 << 22)
        writer.write(head + request)
        await writer.drain()
        status_line = await reader.readline()
        status = int(status_line.split()[1])
        while (await reader.readline()) not in (b"\r\n", b""):
            pass
        if status != 200:
            await reader.read()
            record.update(status=status, error=True)
            writer.close()
            return
        while True:
            line = await reader.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if choice.get("delta", {}).get("content"):
                    times.append(time.perf_counter())
                finish = choice.get("finish_reason") or finish
        writer.close()
    except (OSError, ValueError, IndexError) as error:
        record.update(status=0, error=True, detail=repr(error)[:120])
        return
    if not times or not usage:
        record.update(status=200, error=True, detail="no tokens")
        return
    completion = usage["completion_tokens"]
    record.update(
        status=200, error=False, finish=finish, start=started_at,
        prompt_tokens=usage["prompt_tokens"], completion_tokens=completion,
        ttft=times[0] - started_at, e2e=times[-1] - started_at,
        tpot=(times[-1] - times[0]) / max(completion - 1, 1),
        gaps=[b - a for a, b in zip(times, times[1:])],
    )


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(int(q * len(values)), len(values) - 1)]


def summarize(rows, wall, config):
    good = [row for row in rows if not row.get("error")]
    gaps = [gap for row in good for gap in row["gaps"]]
    summary = {
        "config": config, "requests": len(rows), "ok": len(good),
        "errors": len(rows) - len(good),
        "rejected_429": sum(row.get("status") == 429 for row in rows),
        "other_failures": [row.get("detail") or row.get("status") for row in rows if row.get("error") and row.get("status") != 429][:4],
        "wall_s": wall,
        "output_tok_s": sum(row["completion_tokens"] for row in good) / wall if wall else 0,
        "requests_s": len(good) / wall if wall else 0,
        "prompt_tokens_mean": sum(row["prompt_tokens"] for row in good) / max(len(good), 1),
        "completion_tokens_mean": sum(row["completion_tokens"] for row in good) / max(len(good), 1),
    }
    for name in ("ttft", "tpot", "e2e"):
        values = [row[name] * 1000 for row in good]
        for q, label in ((.5, "p50"), (.95, "p95"), (.99, "p99")):
            summary[f"{name}_{label}_ms"] = percentile(values, q)
    summary["itl_p99_ms"] = (percentile(gaps, .99) or 0) * 1000
    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("--prompt-words", default="30")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--burst")
    parser.add_argument("--requests", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out")
    parser.add_argument("--extra", default="{}")
    parser.add_argument("--records", help="also write each measured request's start, TTFT, E2E and prompt tokens")
    parser.add_argument("--shared-prefix-words", type=int, default=0,
                        help="start every request with the same system document of N words")
    args = parser.parse_args()
    host, port = args.address.split(":")
    rng = random.Random(args.seed)
    lengths = [int(value) for value in args.prompt_words.split(",")]
    total = args.warmup + args.requests
    shared = []
    if args.shared_prefix_words:
        shared = [{"role": "system", "content": "Reference:\n\n" + document(random.Random(0), args.shared_prefix_words)}]
    bodies = [{
        "model": args.model, "temperature": 0, "max_tokens": args.max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
        "messages": shared + [{"role": "user", "content": prompt(rng, rng.choice(lengths))}],
        **json.loads(args.extra),
    } for _ in range(total)]
    records = [{} for _ in range(total)]
    tasks = []
    start = time.perf_counter()

    if args.rate:
        on, off = (0.0, 0.0)
        if args.burst:
            on, off = (float(value) for value in args.burst.split(","))
        peak = args.rate * ((on + off) / on if on else 1)
        clock = 0.0
        for index in range(total):
            clock += rng.expovariate(peak)
            arrival = clock
            if on:
                cycles = int(clock // on)
                arrival = cycles * (on + off) + (clock - cycles * on)
            delay = start + arrival - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            at = time.perf_counter()
            tasks.append(asyncio.create_task(one(host, int(port), bodies[index], at, records[index])))
        await asyncio.gather(*tasks)
    else:
        queue = iter(range(total))

        async def worker():
            for index in queue:
                await one(host, int(port), bodies[index], time.perf_counter(), records[index])

        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    wall = time.perf_counter() - start
    measured = records[args.warmup:]
    summary = summarize(measured, wall * len(measured) / total, vars(args))
    print(json.dumps({key: (round(value, 2) if isinstance(value, float) else value)
                      for key, value in summary.items() if key != "config"}))
    if args.out:
        with open(args.out, "w") as output:
            json.dump(summary, output, indent=1)
    if args.records:
        with open(args.records, "w") as output:
            json.dump([{"start": row["start"] - start, "ttft": row["ttft"], "e2e": row["e2e"],
                        "prompt_tokens": row["prompt_tokens"]} for row in measured if not row.get("error")],
                      output)


asyncio.run(main())
