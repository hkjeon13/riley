#!/usr/bin/env python3
"""Record log probabilities, concurrently, for byte comparison across two server builds.

usage: logprob_parity.py ADDRESS MODEL OUT.json [--against OTHER.json]
Mixes completions and chat, top_logprobs 0/1/5/20, and requests without
logprobs in the same batches. Compares the raw response text of each choice's
content and logprobs, so any changed digit counts.
"""
import argparse
import concurrent.futures
import http.client
import json

PROMPTS = ["The history of the printing press", "How a refrigerator works", "Write a haiku about rain",
           "List three prime numbers and explain why", "Describe the city of Busan", "What is a monad?",
           "Explain photosynthesis to a child", "A short story about a lighthouse keeper"]


def call(address, model, prompt, top, chat):
    host, port = address.split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=300)
    body = {"model": model, "temperature": 0, "max_tokens": 48}
    if chat:
        body["messages"] = [{"role": "user", "content": prompt}]
        if top is not None:
            body |= {"logprobs": True, "top_logprobs": top}
        path = "/v1/chat/completions"
    else:
        body["prompt"] = prompt + ":"
        if top is not None:
            body["logprobs"] = min(top, 5)
        path = "/v1/completions"
    connection.request("POST", path, json.dumps(body), {"content-type": "application/json"})
    reply = json.loads(connection.getresponse().read())
    if "choices" not in reply:
        raise SystemExit(f"{prompt!r}: {reply}")
    choice = reply["choices"][0]
    return json.dumps({key: choice.get(key) for key in ("text", "message", "logprobs", "finish_reason")}, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--against")
    args = parser.parse_args()
    jobs = [(p, top, chat) for p in PROMPTS for top in (None, 0, 1, 5, 20) for chat in (False, True)]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda job: call(args.address, args.model, *job), jobs))
    json.dump(replies, open(args.out, "w"), indent=1)
    if args.against:
        other = json.load(open(args.against))
        same = sum(a == b for a, b in zip(replies, other))
        with_logprobs = sum('"logprobs": null' not in a for a in replies)
        print(json.dumps({"requests": len(replies), "identical": same, "with_logprobs": with_logprobs}))


main()
