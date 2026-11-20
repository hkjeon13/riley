#!/usr/bin/env python3
"""Witness committed mixed prefill/decode on a dedicated Riley server."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from http.client import HTTPConnection
import json
from pathlib import Path
from threading import Event
import time

from long_accuracy import token_ids


CHUNK = 256
REPEATS = 3
PROMPTS = {
    "decode": ("Continue the numbered list with hundreds of detailed lighthouse restoration instructions.\n"
               "1. Record the dimensions and location of every window.\n2."),
    "prefill": ("The harbor engineer recorded the depth, current, and stone condition every morning.\n" * 220
                + "\nSummarize the repeated observation in one paragraph.\nSummary:"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def capture(address, model, name, generated=None):
    limit = 256 if name == "decode" else 64
    request = {"model": model, "prompt": PROMPTS[name], "temperature": 0, "max_tokens": limit,
               "return_token_ids": True, "stream": True, "stream_options": {"include_usage": True}}
    connection = HTTPConnection(address, timeout=120)
    prompt, usage, finish = None, None, None
    tokens, text = [], []
    done = False
    try:
        connection.request("POST", "/v1/completions", json.dumps(request),
                           {"Content-Type": "application/json", "Connection": "close"})
        response = connection.getresponse()
        require(response.status == 200, f"{name}: unexpected HTTP {response.status}")
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            require("error" not in event, f"{name}: SSE error")
            if event.get("usage") is not None:
                require(usage is None, f"{name}: duplicate usage")
                usage = event["usage"]
            for choice in event["choices"]:
                if choice.get("prompt_token_ids") is not None:
                    require(prompt is None, f"{name}: duplicate prompt IDs")
                    prompt = token_ids(choice["prompt_token_ids"])
                row = token_ids(choice.get("token_ids", []))
                require(finish is None or not row and not choice["text"], f"{name}: content after finish")
                tokens.extend(row)
                text.append(choice["text"])
                if generated is not None and len(tokens) >= 8:
                    generated.set()
                if choice["finish_reason"] is not None:
                    require(finish is None, f"{name}: duplicate finish")
                    finish = choice["finish_reason"]
    finally:
        connection.close()
    require(done and prompt and tokens and finish in ("stop", "length") and usage is not None,
            f"{name}: incomplete token stream")
    require(0 < len(tokens) <= limit and usage["prompt_tokens"] == len(prompt)
            and usage["completion_tokens"] in (len(tokens), len(tokens) + int(finish == "stop"))
            and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            and (finish != "length" or len(tokens) == limit), f"{name}: invalid usage")
    return {"prompt": prompt, "tokens": tokens, "text": "".join(text), "finish": finish, "usage": usage}


def observation(address):
    connection = HTTPConnection(address, timeout=5)
    try:
        connection.request("GET", "/metrics", headers={"Connection": "close"})
        response = connection.getresponse()
        require(response.status == 200, "metrics request failed")
        metrics = json.load(response)
    finally:
        connection.close()
    batch = metrics["batch_shapes"]
    require(not batch["metrics_degraded"] and batch["bucket_count"] > 0, "batch-shape metrics unavailable")
    last = batch["last"]
    step = sum(bucket["hit_count"] for bucket in batch["buckets"])
    probe = {"iteration": step, "inflight": metrics["active_requests"],
             "waiting": metrics["waiting_requests"], "kv_blocks": metrics["kv_allocated_blocks"]}
    if last is not None:
        bucket, = [item for item in batch["buckets"] if item["dense_rows"] == last["selected_dense_rows"]]
        require(last["active_rows"] + last["padding_rows"] == last["selected_dense_rows"],
                "inconsistent padded shape")
        probe.update(last, gpu_ns=bucket["gpu_execution_ns_last"])
    return probe


def collect(address, futures):
    samples = {}
    while not all(future.done() for future in futures):
        sample = observation(address)
        if sample.get("active_rows", 0) and sample["iteration"] not in samples:
            samples[sample["iteration"]] = sample
        time.sleep(0.006)
    return samples


def result_digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def probe(address, model):
    initial = observation(address)
    require(initial["inflight"] == initial["waiting"] == initial["kv_blocks"] == 0, "need an idle server")
    baselines, serial = {}, {}
    trials = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for name in PROMPTS:
            future = executor.submit(capture, address, model, name)
            samples = collect(address, (future,))
            baselines[name] = future.result()
            serial[name] = max(sample["active_rows"] for sample in samples.values())
            require(serial[name] <= CHUNK, f"{name}: serial control exceeded the chunk size")
        require(len(baselines["decode"]["tokens"]) >= 64 and len(baselines["decode"]["prompt"]) <= CHUNK,
                "decode control must start shortly and produce at least 64 tokens")
        require(len(baselines["prefill"]["prompt"]) >= 8 * CHUNK and serial["prefill"] == CHUNK,
                "prefill control must have multiple full chunks")
        for _ in range(REPEATS):
            generated = Event()
            seed = executor.submit(capture, address, model, "decode", generated)
            require(generated.wait(timeout=30) and not seed.done(), "decode stream did not reach eight live tokens")
            before = observation(address)
            require(not seed.done() and before["inflight"] == 1 and before["waiting"] == 0
                    and before["active_rows"] == 1 and before["gpu_ns"] > 0, "no committed singleton decode control")
            other = executor.submit(capture, address, model, "prefill")
            samples = collect(address, (seed, other))
            observed = [sample for sample in samples.values()
                        if sample["iteration"] > before["iteration"] and sample["active_rows"] == CHUNK + 1
                        and sample["inflight"] == 2 and sample["gpu_ns"] > 0]
            require(observed, "no committed 257-active-row iteration observed")
            output = {"decode": seed.result(), "prefill": other.result()}
            require(output == baselines, "mixed scheduling changed prompt IDs, generated IDs, text, finish or usage")
            trials.append({"prelaunch": before, "witnesses": observed, "matching_responses": len(output)})
    final = observation(address)
    require(final["inflight"] == final["waiting"] == final["kv_blocks"] == 0, "requests or KV did not drain")
    return {"status": "passed", "prefill_chunk_tokens": CHUNK,
            "serial_max_active_rows": serial,
            "baselines": {name: {"body_sha256": result_digest(PROMPTS[name]), "response_sha256": result_digest(value),
                                 "prompt_tokens": len(value["prompt"]), "generated_ids": len(value["tokens"]),
                                 "finish": value["finish"]} for name, value in baselines.items()},
            "trials": trials, "drained": final}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.address, args.model)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
