import json, random, sys, time, urllib.request, urllib.error
addr = sys.argv[1]
words = "the of model server request token memory context cache vector search document answer query system data result value time long short".split()
def req(n, mt, chat=False):
    rng = random.Random(n); text = " ".join(rng.choice(words) for _ in range(n))
    if chat: body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": mt, "temperature": 0}; path = "chat/completions"
    else: body = {"model": "m", "prompt": text, "max_tokens": mt, "temperature": 0}; path = "completions"
    t = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://{addr}/v1/{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)); out = r["usage"]["prompt_tokens"]
    except urllib.error.HTTPError as e: out = "HTTP%d %s" % (e.code, e.read()[:80])
    return round((time.perf_counter() - t) * 1000, 2), out
for n, mt, chat in [(20, 1, False), (2600, 1, False), (2600, 1, True), (9000, 1, False), (20000, 1, False)]:
    print(n, mt, chat, [req(n, mt, chat) for _ in range(4)])
