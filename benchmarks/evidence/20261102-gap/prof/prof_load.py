#!/usr/bin/env python3
"""prof_load.py HOST:PORT MODEL [REQUESTS]: sequential 2,600-word prompts, 4 output tokens."""
import json, random, sys, time, urllib.request
host, model = sys.argv[1:3]
requests = int(sys.argv[3]) if len(sys.argv) > 3 else 8
words = "the of model server request token memory context cache vector search document answer query system data result value time long short".split()
for seed in range(requests):
    rng = random.Random(seed)
    text = " ".join(rng.choice(words) for _ in range(2600))
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 4, "temperature": 0}).encode()
    start = time.time()
    req = urllib.request.Request(f"http://{host}/v1/chat/completions", body, {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(req, timeout=120))
    print(seed, reply["usage"]["prompt_tokens"], round(time.time() - start, 3), flush=True)
    time.sleep(0.3)
