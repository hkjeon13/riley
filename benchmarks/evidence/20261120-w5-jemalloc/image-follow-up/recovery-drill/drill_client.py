"""Run the API contract, check a cached prompt, and drain the isolated route."""

from argparse import ArgumentParser
import asyncio
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from load import one


GAUGES = ("active_requests", "waiting_requests", "kv_allocated_blocks")


def fetch(host, port, resource, payload=None):
    wire = HTTPConnection(host, port, timeout=120)
    try:
        if payload is None:
            wire.request("GET", resource)
        else:
            wire.request("POST", resource, body=json.dumps(payload), headers={"Content-Type": "application/json"})
        result = wire.getresponse()
        if result.status != 200:
            raise RuntimeError(f"{resource} returned HTTP {result.status}")
        return json.load(result)
    finally:
        wire.close()


async def stream(host, port, body):
    record = {}
    await asyncio.wait_for(one(host, port, {**body, "stream": True, "stream_options": {"include_usage": True}},
                               time.perf_counter(), record, capture_hashes=True), timeout=120)
    if record.get("error") is not False:
        raise RuntimeError("recovery stream lacks content, usage or a valid terminal event")
    return {key: record[key] for key in ("input_sha256", "output_sha256", "prompt_tokens", "completion_tokens", "finish")}


def run(address, model, output):
    host, raw_port = address.split(":")
    if host != "127.0.0.1":
        raise ValueError("recovery probe requires loopback")
    port = int(raw_port)
    subprocess.run([sys.executable, "-B", os.environ["RILEY_W5_DRILL_API"], address, model,
                    "--tools", "--out", str(output)], check=True, timeout=600)
    prefix = "A Riley recovery drill verifies the same assistant on the restored image. " * 8
    chat = {"model": model, "messages": [{"role": "system", "content": prefix},
                                       {"role": "user", "content": "Reply with one short sentence."}],
            "temperature": 0, "max_tokens": 24}
    for _ in range(2):
        reply = fetch(host, port, "/v1/chat/completions", chat)
    cached = reply["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if type(cached) is not int or cached <= 0:
        raise RuntimeError("cache-on route did not reuse the probe prompt")
    # Keep the cross-image SSE probe independent of the cache hit check.
    body = {"model": model, "messages": [{"role": "user", "content": "Write three words about a maple tree."}],
            "temperature": 0, "max_tokens": 12}
    exact = asyncio.run(stream(host, port, body))
    gated = time.monotonic_ns()
    while True:
        metrics = fetch(host, port, "/metrics")
        if all(type(metrics.get(key)) is int and metrics[key] == 0 for key in GAUGES):
            break
        if time.monotonic_ns() - gated >= 15 * 10**9:
            raise TimeoutError("test client gated new requests but the route did not drain")
        time.sleep(0.1)
    with output.with_name("route-probe.json").open("x", encoding="utf-8") as out:
        json.dump({"endpoint": address, "client_gated_ns": gated, "drained_ns": time.monotonic_ns(),
                   "cached_prompt_tokens": cached, "restored_stream": exact,
                   "drained": {key: metrics[key] for key in GAUGES}}, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("--tools", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.address, args.model, args.out)
