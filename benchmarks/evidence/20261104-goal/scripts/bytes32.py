#!/usr/bin/env python3
"""bytes32.py HOST:PORT OUT: 32 short greedy prompts at once, 512 tokens each (32-row decode steps), texts saved to OUT."""
import json, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
load = type(sys)("load"); exec(open("src/benchmarks/live/load.py").read().split("\nasync def one")[0], load.__dict__)
addr, out = sys.argv[1], sys.argv[2]
def one(i):
    rng = random.Random(500 + i)
    body = json.dumps({"model": "m", "prompt": load.prompt(rng, 30), "max_tokens": 512, "temperature": 0, "ignore_eos": True}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://{addr}/v1/completions", body, {"Content-Type": "application/json"}), timeout=600))
    return [r["usage"]["completion_tokens"], r["choices"][0]["text"]]
with ThreadPoolExecutor(32) as ex: res = list(ex.map(one, range(32)))
json.dump(res, open(out, "w"))
print("completion tokens", sum(c for c, _ in res))
