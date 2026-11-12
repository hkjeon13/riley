#!/usr/bin/env python3
"""bytes.py HOST:PORT OUT: 16 long greedy prompts (8 at a time), texts saved to OUT for byte comparison."""
import json, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
load = type(sys)("load"); exec(open("src/benchmarks/live/load.py").read().split("\nasync def one")[0], load.__dict__)
addr, out = sys.argv[1], sys.argv[2]
def one(i):
    rng = random.Random(100 + i); words = rng.randint(1200, 3200)
    body = json.dumps({"model": "m", "prompt": load.prompt(rng, words), "max_tokens": 64, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://{addr}/v1/completions", body, {"Content-Type": "application/json"}), timeout=600))
    return [r["usage"]["prompt_tokens"], r["choices"][0]["text"]]
with ThreadPoolExecutor(8) as ex: res = list(ex.map(one, range(16)))
json.dump(res, open(out, "w"))
print("prompt tokens", [p for p, _ in res])
