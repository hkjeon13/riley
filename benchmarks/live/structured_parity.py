#!/usr/bin/env python3
"""Record structured completions, concurrently, for comparison across two server builds.

usage: structured_parity.py ADDRESS MODEL OUT.json [--against OTHER.json]
"""
import argparse
import concurrent.futures
import http.client
import json

TOPICS = ["a novel about the sea", "a textbook on geology", "a history of Korea", "a cookbook for winter",
          "a guide to Rust", "a memoir by a pilot", "a survey of bridges", "a book of chess problems"]
SCHEMAS = [
    {"type": "object", "properties": {"title": {"type": "string"}, "chapters": {"type": "array", "items": {"type": "string"}},
                                      "year": {"type": "integer"}}, "required": ["title", "chapters", "year"]},
    {"type": "object", "properties": {"genre": {"enum": ["fiction", "reference", "other"]}, "pages": {"type": "integer", "minimum": 1, "maximum": 9999},
                                      "summary": {"type": "string", "maxLength": 80}}, "required": ["genre", "pages", "summary"]},
]


def call(address, model, topic, schema, top):
    host, port = address.split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=120)
    body = {"model": model, "messages": [{"role": "user", "content": f"Describe {topic} as JSON."}], "temperature": 0,
            "max_tokens": 64, "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}}
    if top:
        body |= {"logprobs": True, "top_logprobs": 3}
    connection.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    reply = json.loads(connection.getresponse().read())
    if "choices" not in reply:
        raise SystemExit(f"{topic!r}: {reply}")
    choice = reply["choices"][0]
    return choice["message"]["content"], choice["finish_reason"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--against")
    args = parser.parse_args()
    jobs = [(t, s, top) for t in TOPICS for s in SCHEMAS for top in (False, True)]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda job: call(args.address, args.model, *job), jobs))
    json.dump(replies, open(args.out, "w"), indent=1)
    if args.against:
        other = [tuple(r) for r in json.load(open(args.against))]
        same = sum(a == b for a, b in zip(replies, other))
        print(json.dumps({"requests": len(replies), "identical": same}))


main()
