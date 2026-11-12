#!/usr/bin/env python3
"""reason_client.py ADDRESS: 8 concurrent greedy completions of up to 1,024 tokens, as reason-c8 does; prints tokens."""
import concurrent.futures, json, sys, urllib.request
TOPICS = ["the history of bridges", "how compilers work", "the ocean currents", "a city at night",
          "the life of bees", "trains in winter", "the first computers", "a mountain village"]
def call(topic):
    body = json.dumps({"model": "m", "prompt": f"Write a very long and detailed essay about {topic}.\n\n", "max_tokens": 1024,
                       "temperature": 0}).encode()
    request = urllib.request.Request(f"http://{sys.argv[1]}/v1/completions", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=900))["usage"]["completion_tokens"]
with concurrent.futures.ThreadPoolExecutor(8) as pool:
    print("completion tokens", list(pool.map(call, TOPICS)))
