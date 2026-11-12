#!/usr/bin/env python3
"""bucket_probe.py HOST:PORT: send one prompt per remainder size and report which dense-row bucket ran each prompt step."""
import json, sys, urllib.request
addr = sys.argv[1]
def metrics():
    m = json.load(urllib.request.urlopen(f"http://{addr}/metrics"))
    return {b["dense_rows"]: b["hit_count"] for b in m["batch_shapes"]["buckets"]}
def send(n):
    body = json.dumps({"model": "m", "prompt": "hello" + " hello" * (n - 1), "max_tokens": 1, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://{addr}/v1/completions", body, {"Content-Type": "application/json"})))
    return r["usage"]["prompt_tokens"]
out = []
for rows in [200, 300, 400, 480, 560, 620, 700, 760, 830, 900, 960]:
    before = metrics(); tokens = send(rows); after = metrics()
    hits = {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)}
    out.append(f"{tokens}->{sorted(hits)}")
print(" ".join(out))
