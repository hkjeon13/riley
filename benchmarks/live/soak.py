#!/usr/bin/env python3
"""Mixed open-loop soak: plain, schema, logprobs, tools, long prompts, cancels.

usage: soak.py HOST:PORT MODEL --seconds S --rate R --pid SERVER_PID --out FILE
Every finished schema or tool reply is validated. Afterwards the server must be
idle with its KV blocks returned, and its RSS must not have kept growing.
"""
import argparse
import asyncio
import json
import random
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("address")
parser.add_argument("model")
parser.add_argument("--seconds", type=float, default=1800)
parser.add_argument("--rate", type=float, default=8)
parser.add_argument("--pid", type=int, required=True)
parser.add_argument("--out")
args = parser.parse_args()
HOST, PORT = args.address.split(":")
WORDS = "the river market station garden winter engine letter mountain copper signal harbor forest".split()
SCHEMA = {"type": "object", "properties": {"title": {"type": "string", "maxLength": 40},
                                           "year": {"type": "integer", "minimum": 1000, "maximum": 2100},
                                           "tags": {"type": "array", "items": {"enum": ["a", "b", "c"]}, "maxItems": 3}},
          "required": ["title", "year"], "additionalProperties": False}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string", "maxLength": 30}}, "required": ["city"]}}}]
rng = random.Random(11)
counts = {}
problems = []


def text(words):
    return " ".join(rng.choice(WORDS) for _ in range(words))


def request():
    kind = rng.choices(["plain", "schema", "logprobs", "tools", "long", "reason", "cancel"],
                       [55, 12, 8, 5, 10, 5, 5])[0]
    body = {"model": args.model, "temperature": 0, "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": f"Write about {text(rng.randint(3, 40))}."}]}
    if kind == "schema":
        body.update(max_tokens=96, response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": SCHEMA}})
    elif kind == "logprobs":
        body.update(logprobs=True, top_logprobs=3)
    elif kind == "tools":
        body.update(tools=TOOLS, tool_choice="required", max_tokens=64)
    elif kind == "long":
        body["messages"][0]["content"] = f"{text(rng.choice([800, 1500, 3000]))}\nSummarize the text above."
        body["max_tokens"] = 96
    elif kind == "reason":
        body["max_tokens"] = 512
    return kind, body


async def one(kind, body):
    payload = json.dumps(body).encode()
    head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {HOST}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode()
    counts[kind] = counts.get(kind, 0) + 1
    try:
        reader, writer = await asyncio.open_connection(HOST, int(PORT), limit=1 << 22)
        writer.write(head + payload)
        await writer.drain()
        status = int((await reader.readline()).split()[1])
        while (await reader.readline()) not in (b"\r\n", b""):
            pass
        if status != 200:
            counts[f"http{status}"] = counts.get(f"http{status}", 0) + 1
            if status != 429:
                problems.append((kind, status))
            writer.close()
            return
        content, calls, finish, chunks = "", [], None, 0
        while True:
            line = await reader.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                problems.append((kind, chunk["error"]))
            for choice in chunk.get("choices") or []:
                content += choice.get("delta", {}).get("content") or ""
                calls += choice.get("delta", {}).get("tool_calls") or []
                finish = choice.get("finish_reason") or finish
            chunks += 1
            if kind == "cancel" and chunks == 5:
                writer.close()
                return
        writer.close()
        if kind == "schema" and finish == "stop":
            value = json.loads(content)
            if not (set(value) <= {"title", "year", "tags"} and 1000 <= value["year"] <= 2100):
                problems.append((kind, content))
        if kind == "tools" and finish == "tool_calls":
            json.loads(calls[0]["function"]["arguments"])
        elif kind == "tools" and finish == "stop":
            problems.append((kind, "forced tool call finished without a call", content[:80]))
    except (OSError, ValueError, KeyError, IndexError) as error:
        problems.append((kind, repr(error)[:160]))


def rss_mb():
    with open(f"/proc/{args.pid}/status") as status:
        for line in status:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


def health():
    return json.load(urllib.request.urlopen(f"http://{args.address}/readyz", timeout=10))


def metrics():
    return json.load(urllib.request.urlopen(f"http://{args.address}/metrics", timeout=10))


async def main():
    start = time.perf_counter()
    tasks, samples, next_sample = [], [], 0.0
    clock = 0.0
    while clock < args.seconds:
        clock += rng.expovariate(args.rate)
        delay = start + clock - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        kind, body = request()
        tasks.append(asyncio.create_task(one(kind, body)))
        if clock >= next_sample:
            samples.append((round(clock), round(rss_mb(), 1)))
            print(samples[-1], len(tasks), len(problems), flush=True)
            next_sample += 60
            tasks = [task for task in tasks if not task.done()]
    await asyncio.gather(*tasks)
    await asyncio.sleep(5)
    final = metrics()
    tail = [value for _, value in samples[len(samples) // 2:]]
    result = {"seconds": args.seconds, "rate": args.rate, "counts": counts, "problems": problems[:20],
              "problem_count": len(problems), "rss_samples": samples, "rss_growth_second_half_mb": tail[-1] - tail[0],
              "final_health": health(), "kv_allocated_blocks": final.get("kv_allocated_blocks"),
              "counters": final.get("counters")}
    print(json.dumps({key: value for key, value in result.items() if key != "rss_samples"})[:1500])
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)


asyncio.run(main())
