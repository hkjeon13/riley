#!/usr/bin/env python3
"""Check prefix reuse, eviction and 429 recovery on a dedicated Riley server."""

import argparse
import asyncio
from hashlib import sha256
import json
import random
import time
import urllib.request

from load import document
from soak import idle, run_request


def request(model, content, *, stream=False, limit=1):
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": limit, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def read_json(address, path, body=None):
    req = urllib.request.Request(f"http://{address}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def complete(address, body):
    payload = json.dumps(body).encode()
    result = read_json(address, "/v1/chat/completions", payload)
    choice, = result["choices"]
    usage = result["usage"]
    prompt_tokens = usage["prompt_tokens"]
    cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if (choice["finish_reason"] not in ("stop", "length")
            or type(prompt_tokens) is not int or type(cached_tokens) is not int
            or not 0 <= cached_tokens < prompt_tokens or cached_tokens % 16
            or usage["completion_tokens"] != 1 or usage["total_tokens"] != prompt_tokens + 1):
        raise ValueError("invalid one-token completion or cache usage")
    return {"prompt_sha256": sha256(payload).hexdigest(),
            "visible_output_sha256": sha256((choice["message"]["content"] or "").encode()).hexdigest(),
            "finish": choice["finish_reason"], "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens}


async def burst(address, model):
    host, port = address.split(":")
    body = request(model, "Write a long story about a lighthouse.", stream=True, limit=512)
    scheduled = time.perf_counter()
    return await asyncio.gather(*(run_request(host, int(port), "plain", body, scheduled) for _ in range(16)))


async def wait_idle(address):
    deadline = time.monotonic() + 10
    while True:
        metrics = await asyncio.to_thread(read_json, address, "/metrics")
        if idle(metrics) or time.monotonic() >= deadline:
            return metrics
        await asyncio.sleep(.1)


async def check(args):
    before = await wait_idle(args.address)
    if not idle(before):
        raise ValueError("requires a dedicated idle server")
    a = request(args.model, "Harbor notes\n" + document(random.Random(29), 1200))
    b = request(args.model, "Glacier report\n" + document(random.Random(31), 2400))
    rows = {}
    for name, body in (("cold", a), ("warm", a), ("pressure", b), ("after_pressure", a), ("rewarmed", a)):
        rows[name] = await asyncio.to_thread(complete, args.address, body)

    cold, warm, pressure, after, rewarmed = rows.values()
    cache_tokens = args.prefix_cache_blocks * 16
    matched = await wait_idle(args.address)
    arrivals = await burst(args.address, args.model)
    drained = await wait_idle(args.address)
    recovered = await asyncio.to_thread(complete, args.address, a)
    final = await wait_idle(args.address)
    overloads = drained["counters"]["overloads"] - matched["counters"]["overloads"]
    rejected = sum(row.get("http_status") == 429 for row in arrivals)
    completed = sum(row["outcome"] == "completed" for row in arrivals)
    checks = {
        "cache_pressure_exceeds_capacity": cold["prompt_tokens"] <= cache_tokens < pressure["prompt_tokens"] - 16,
        "cold_and_evicted_prefix_miss": cold["cached_tokens"] == after["cached_tokens"] == 0,
        "warm_prefix_hit": warm["cached_tokens"] == rewarmed["cached_tokens"]
        == ((warm["prompt_tokens"] - 1) // 16) * 16 > 0,
        "output_stable": all((row["visible_output_sha256"], row["finish"], row["prompt_tokens"])
                             == (cold["visible_output_sha256"], cold["finish"], cold["prompt_tokens"])
                             for row in (warm, after, rewarmed, recovered)),
        "saturation_returns_only_expected_429": 0 < rejected == overloads and completed > 0
        and completed + rejected == len(arrivals),
        "drain_and_recovery": idle(matched) and idle(drained) and idle(final)
        and final["counters"]["overloads"] == drained["counters"]["overloads"]
        and final["allocation"] == before["allocation"],
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "prefix_cache_blocks": args.prefix_cache_blocks, "prefix_requests": rows,
            "overload": {"arrivals": arrivals, "completed": completed, "rejected": rejected,
                         "counter_delta": overloads, "recovery": recovered},
            "metrics": {"before": before, "before_burst": matched, "drained": drained, "final": final}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("--prefix-cache-blocks", type=int, default=128)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.prefix_cache_blocks <= 0:
        parser.error("--prefix-cache-blocks must be positive")
    result = asyncio.run(check(args))
    with open(args.out, "x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(result["checks"]))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
