#!/usr/bin/env python3
"""Live API contract check against a running Riley server (stdlib only).

usage: api_check.py HOST:PORT MODEL_ID [--key KEY] [--tools] [--out result.json]
Exits non-zero if any check fails.
"""
import argparse
import concurrent.futures as futures
import http.client
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument("address")
parser.add_argument("model")
parser.add_argument("--key")
parser.add_argument("--tools", action="store_true")
parser.add_argument("--out")
args = parser.parse_args()
HOST, PORT = args.address.split(":")
RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": str(detail)[:400]})
    print(("PASS " if passed else "FAIL ") + name + ("" if passed else f"  {str(detail)[:300]}"), flush=True)


def post(path, body, key=args.key):
    connection = http.client.HTTPConnection(HOST, int(PORT), timeout=300)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    connection.request("POST", path, json.dumps(body), headers)
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data


def chat(**fields):
    body = {"model": args.model, "max_tokens": 64, "temperature": 0, **fields}
    status, data = post("/v1/chat/completions", body)
    try:
        return status, json.loads(data)
    except ValueError:
        return status, {"raw": data[:300].decode(errors="replace")}


def stream(path, body):
    connection = http.client.HTTPConnection(HOST, int(PORT), timeout=300)
    headers = {"Content-Type": "application/json"}
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"
    connection.request("POST", path, json.dumps({**body, "stream": True}), headers)
    response = connection.getresponse()
    chunks = []
    for line in response.read().decode().split("\n\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    connection.close()
    return response.status, chunks


def user(text):
    return [{"role": "user", "content": text}]


PROMPTS = [
    "Describe a city in Korea.", "Invent a fictional book.", "Summarize the water cycle.",
    "Name a programming language and its creator.", "Give a recipe for tea.",
    "Describe the planet Mars.", "Name three colors.", "Who wrote Hamlet?",
]
PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 40},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "kind": {"enum": ["human", "robot"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "note": {"type": ["string", "null"], "maxLength": 60},
    },
    "required": ["name", "age", "kind"],
    "additionalProperties": False,
}


def valid_person(value):
    return (isinstance(value, dict)
            and set(value) <= set(PERSON["properties"])
            and {"name", "age", "kind"} <= set(value)
            and 1 <= len(value["name"]) <= 40
            and isinstance(value["age"], int) and 0 <= value["age"] <= 150
            and value["kind"] in ("human", "robot")
            and len(value.get("tags", [])) <= 3)


# Authentication.
if args.key:
    status, _ = post("/v1/chat/completions", {"model": args.model, "messages": user("hi")}, key="wrong")
    check("auth: wrong key is 401", status == 401, status)
    status, _ = post("/v1/chat/completions", {"model": args.model, "messages": user("hi")}, key=None)
    check("auth: missing key is 401", status == 401, status)

# Plain request, baseline for invariance.
status, plain = chat(messages=user(PROMPTS[0]))
check("plain chat", status == 200 and plain["choices"][0]["message"]["content"], plain)
status, body = post("/v1/completions", {"model": args.model, "prompt": "Hello", "max_tokens": 4,
                                        "user": "someone", "temperature": 0})
check("completions accepts user", status == 200, body[:200])

# Log probabilities.
status, reply = chat(messages=user(PROMPTS[1]), logprobs=True, top_logprobs=5, max_tokens=24)
content = reply.get("choices", [{}])[0].get("logprobs", {}) or {}
content = content.get("content", [])
check("chat logprobs present", status == 200 and len(content) == reply["usage"]["completion_tokens"], reply)
if content:
    check("greedy token is the most likely",
          all(entry["top_logprobs"][0]["token"] == entry["token"]
              and abs(entry["top_logprobs"][0]["logprob"] - entry["logprob"]) < 1e-5
              for entry in content), content[:2])
    check("top_logprobs sorted, five each, negative",
          all(len(entry["top_logprobs"]) == 5
              and all(a["logprob"] >= b["logprob"] for a, b in zip(entry["top_logprobs"], entry["top_logprobs"][1:]))
              and entry["logprob"] <= 0 for entry in content), content[:1])
    joined = bytes(byte for entry in content for byte in entry["bytes"]).decode(errors="replace")
    text = reply["choices"][0]["message"]["content"]
    check("token bytes rebuild the text", joined.startswith(text), (joined, text))
    check("top mass at most one",
          all(sum(2.718281828 ** t["logprob"] for t in entry["top_logprobs"]) <= 1.0001 for entry in content))
status, body = post("/v1/completions", {"model": args.model, "prompt": "The capital of France is",
                                        "max_tokens": 5, "logprobs": 2, "temperature": 0})
legacy = json.loads(body)["choices"][0].get("logprobs") or {}
check("legacy logprobs shape",
      status == 200 and len(legacy.get("tokens", [])) == 5 and len(legacy["token_logprobs"]) == 5
      and all(len(top) == 2 for top in legacy["top_logprobs"]) and legacy["text_offset"][0] == 0, body[:300])
status, reply = chat(messages=user(PROMPTS[1]), logprobs=True, temperature=0.8, seed=7, max_tokens=16)
check("sampled logprobs", status == 200 and len(reply["choices"][0]["logprobs"]["content"]) > 0
      and "token_ids" not in reply["choices"][0], reply)
status, chunks = stream("/v1/chat/completions", {"model": args.model, "messages": user(PROMPTS[1]),
                                                 "logprobs": True, "top_logprobs": 2, "max_tokens": 16,
                                                 "temperature": 0, "stream_options": {"include_usage": True}})
entries = [e for c in chunks for ch in c["choices"] for e in ((ch.get("logprobs") or {}).get("content") or [])]
usage = [c for c in chunks if c.get("usage")]
check("streamed logprobs", status == 200 and len(entries) == usage[0]["usage"]["completion_tokens"]
      and all(c["object"] == "chat.completion.chunk" for c in chunks), chunks[-3:])

# Structured output.
for index, prompt in enumerate(PROMPTS):
    status, reply = chat(messages=user(prompt + " Answer as a JSON object."), max_tokens=128,
                         response_format={"type": "json_object"})
    choice = reply.get("choices", [{}])[0]
    finished = choice.get("finish_reason") == "stop"
    try:
        parsed = json.loads(choice["message"]["content"])
        good = isinstance(parsed, dict)
    except (KeyError, ValueError):
        good = False
    check(f"json_object {index}", status == 200 and (good or not finished), choice)
person_format = {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON, "strict": True}}
finished_count = 0
for index, prompt in enumerate(PROMPTS):
    temperature = 0 if index % 2 == 0 else 1.0
    status, reply = chat(messages=user(f"Invent a character. Topic: {prompt}"), max_tokens=128,
                         temperature=temperature, seed=index, response_format=person_format)
    choice = reply.get("choices", [{}])[0]
    finished = choice.get("finish_reason") == "stop"
    finished_count += finished
    try:
        good = valid_person(json.loads(choice["message"]["content"]))
    except (KeyError, ValueError):
        good = False
    check(f"json_schema {index}", status == 200 and (good or not finished), choice)
check("most schema outputs finish inside max_tokens", finished_count >= 6, finished_count)
status, chunks = stream("/v1/chat/completions", {"model": args.model, "messages": user("Invent a character."),
                                                 "response_format": person_format, "max_tokens": 128, "temperature": 0})
text = "".join(ch["delta"].get("content", "") for c in chunks for ch in c["choices"])
try:
    streamed_good = valid_person(json.loads(text))
except ValueError:
    streamed_good = False
check("streamed json_schema", status == 200 and streamed_good, text)
status, reply = chat(messages=user("x"), response_format={"type": "json_schema", "json_schema": {
    "name": "n", "schema": {"type": "array", "uniqueItems": True}}})
check("unenforceable schema is 400", status == 400 and "uniqueItems" in json.dumps(reply), reply)
status, reply = chat(messages=user("Give me a phone number."), max_tokens=32, response_format={"type": "json_schema", "json_schema": {
    "name": "n", "schema": {"type": "object", "properties": {"phone": {"type": "string", "pattern": "^\\d{3}-\\d{4}$"}}, "required": ["phone"]}}})
import re
check("pattern is enforced", status == 200 and re.fullmatch(r"\d{3}-\d{4}", json.loads(reply["choices"][0]["message"]["content"])["phone"]), reply)
status, reply = chat(messages=user("Say hi."), response_format={"type": "text"})
check("response_format text", status == 200, reply)

# Other requests must not change while constrained work shares their batches.
def plain_text(prompt):
    return chat(messages=user(prompt), max_tokens=48)[1]["choices"][0]["message"]["content"]

alone = {prompt: plain_text(prompt) for prompt in PROMPTS}
with futures.ThreadPoolExecutor(16) as pool:
    mixed = []
    for _ in range(3):
        for prompt in PROMPTS:
            mixed.append((prompt, pool.submit(plain_text, prompt)))
            pool.submit(chat, messages=user(prompt), logprobs=True, max_tokens=32)
            pool.submit(chat, messages=user(prompt), response_format=person_format, max_tokens=48)
    differing = [(prompt, future.result(), alone[prompt]) for prompt, future in mixed
                 if future.result() != alone[prompt]]
check("mixed batches leave plain outputs identical", not differing, differing[:1])

# Constrained output itself must not depend on its neighbours.
def schema_text(prompt):
    return chat(messages=user(prompt), response_format=person_format, max_tokens=64)[1]["choices"][0]["message"]["content"]

alone = {prompt: schema_text(prompt) for prompt in PROMPTS}
with futures.ThreadPoolExecutor(16) as pool:
    jobs = [(prompt, pool.submit(schema_text, prompt)) for _ in range(3) for prompt in PROMPTS]
    differing = [(p, f.result(), alone[p]) for p, f in jobs if f.result() != alone[p]]
check("schema output is concurrency-invariant", not differing, differing[:1])

if args.tools:
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
        {"type": "function", "function": {
            "name": "add", "description": "Add two integers",
            "parameters": {"type": "object", "properties": {
                "a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]
    question = user("What is the weather in Seoul right now?")
    status, reply = chat(messages=question, tools=tools, tool_choice="required", max_tokens=96)
    choice = reply.get("choices", [{}])[0]
    calls = choice.get("message", {}).get("tool_calls") or []
    check("tool_choice required", status == 200 and calls and choice["finish_reason"] == "tool_calls", reply)
    if calls:
        arguments = json.loads(calls[0]["function"]["arguments"])
        check("required call has valid arguments", calls[0]["function"]["name"] in ("get_weather", "add")
              and isinstance(arguments, dict), calls)
    status, reply = chat(messages=question, tools=tools, max_tokens=96,
                         tool_choice={"type": "function", "function": {"name": "add"}})
    calls = reply.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
    check("named tool_choice", status == 200 and calls and calls[0]["function"]["name"] == "add"
          and {"a", "b"} <= set(json.loads(calls[0]["function"]["arguments"])), reply)
    status, reply = chat(messages=question, tools=tools, max_tokens=128)
    choice = reply.get("choices", [{}])[0]
    check("auto tool call (model's own choice)", status == 200, reply)
    print("  auto ->", json.dumps(choice.get("message"))[:240])
    status, reply = chat(messages=question, tools=tools, tool_choice="none", max_tokens=24)
    check("tool_choice none", status == 200 and not reply["choices"][0]["message"].get("tool_calls"), reply)
    status, chunks = stream("/v1/chat/completions", {
        "model": args.model, "messages": question, "tools": tools, "tool_choice": "required",
        "max_tokens": 96, "temperature": 0, "stream_options": {"include_usage": True}})
    streamed_calls = [call for c in chunks for ch in c["choices"] for call in ch["delta"].get("tool_calls", [])]
    finish = [ch["finish_reason"] for c in chunks for ch in c["choices"] if ch["finish_reason"]]
    check("streamed tool call", status == 200 and streamed_calls and finish == ["tool_calls"], chunks[-2:])
    call = streamed_calls[0] if streamed_calls else {"id": "c", "type": "function",
                                                     "function": {"name": "get_weather", "arguments": "{}"}}
    follow = question + [
        {"role": "assistant", "content": None, "tool_calls": [{key: call[key] for key in ("id", "type", "function")}]},
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"temperature_c": 21, "sky": "clear"})}]
    status, reply = chat(messages=follow, tools=tools, max_tokens=64)
    answer = reply.get("choices", [{}])[0].get("message", {}).get("content") or ""
    check("tool result round trip", status == 200 and "21" in answer, reply)

passed = sum(result["passed"] for result in RESULTS)
print(f"{passed}/{len(RESULTS)} checks passed")
if args.out:
    with open(args.out, "w") as output:
        json.dump({"model": args.model, "at": time.time(), "results": RESULTS}, output, indent=1)
raise SystemExit(passed != len(RESULTS))
