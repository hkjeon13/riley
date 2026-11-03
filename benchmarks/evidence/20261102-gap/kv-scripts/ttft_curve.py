#!/usr/bin/env python3
"""ttft_curve.py HOST:PORT MODEL OUT: one client, TTFT by prompt length, straight-line fit."""
import asyncio, json, random, statistics, sys
import time, types
# load.py parses arguments on import, so take only its prompt helpers.
source = open("src/benchmarks/live/load.py").read().split("async def main")[0]
source = source.split("if __name__")[0]
load = types.ModuleType("load")
exec(source, load.__dict__)
host, port = sys.argv[1].split(":")
model, out = sys.argv[2:4]
WORDS = [50, 200, 400, 800, 1500, 2200, 3000, 4000, 5000]
async def main():
    rng = random.Random(7)
    rows = []
    for words in [400] * 4 + WORDS * 6:
        body = {"model": model, "temperature": 0, "max_tokens": 4, "stream": True, "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": load.prompt(rng, words)}]}
        record = {}
        await load.one(host, int(port), body, time.perf_counter(), record)
        rows.append((words, record.get("prompt_tokens"), record.get("ttft")))
    rows = rows[4:]
    summary = []
    for words in WORDS:
        tokens = [r[1] for r in rows if r[0] == words]
        ttft = [r[2] * 1e3 for r in rows if r[0] == words]
        summary.append({"words": words, "tokens": round(statistics.mean(tokens)), "ttft_ms": round(statistics.median(ttft), 1)})
    x = [s["tokens"] for s in summary]; y = [s["ttft_ms"] for s in summary]
    mx, my = statistics.mean(x), statistics.mean(y)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    result = {"curve": summary, "us_per_token": round(slope * 1e3, 1), "intercept_ms": round(my - slope * mx, 1)}
    json.dump(result, open(out, "w"), indent=1)
    print(json.dumps(result))
asyncio.run(main())
