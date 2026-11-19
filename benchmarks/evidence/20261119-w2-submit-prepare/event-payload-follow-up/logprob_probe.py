"""Preserve the full streamed logprobs and token content as exact fingerprints."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from http.client import HTTPConnection
import json
from pathlib import Path


PROMPTS = (
    "Briefly name three animals that live near rivers.",
    "List three words that rhyme with tree.",
    "서울의 계절을 한 문장으로 설명하세요.",
    "日本語で山について一文を書いてください。",
    "Écrivez une phrase courte sur la mer.",
    "Write three words about balanced parentheses.",
    "Give a short description of a red maple leaf.",
    "Name two fruits and list one color for each.",
)


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def sample(address, prompt):
    host, port = address.split(":")
    if host != "127.0.0.1":
        raise ValueError("probe requires the isolated loopback server")
    payload = {"model": "m", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 24, "stream": True,
               "stream_options": {"include_usage": True}, "logprobs": True, "top_logprobs": 3}
    connection = HTTPConnection(host, int(port), timeout=90)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"stream probe returned HTTP {response.status}")
        chunks = []
        usage = None
        finish = None
        entries = []
        done = False
        for wire in response:
            if not wire.startswith(b"data: "):
                continue
            raw = wire[6:].strip()
            if raw == b"[DONE]":
                done = True
                break
            chunk = json.loads(raw)
            choices = chunk["choices"]
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            if choices:
                choice, = choices
                finish = choice.get("finish_reason") or finish
                entry = choice.get("logprobs")
                if entry:
                    entries.extend(entry["content"])
            chunks.append({key: value for key, value in chunk.items() if key not in ("id", "created")})
        if not (done and usage and finish in ("stop", "length") and entries
                and len(entries) == usage["completion_tokens"] and all(len(entry["top_logprobs"]) == 3 for entry in entries)):
            raise ValueError("incomplete stream, usage, or top logprob rows")
        return {"input_sha256": digest(payload), "stream_sha256": digest(chunks),
                "logprobs_sha256": digest(entries), "finish": finish, "usage": usage, "logprob_rows": len(entries)}
    finally:
        connection.close()


def save_probe(address, output):
    sequential = [sample(address, prompt) for prompt in PROMPTS]
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(lambda prompt: sample(address, prompt), PROMPTS))
    with Path(output).open("x") as destination:
        json.dump({"c1": sequential, "c8": concurrent}, destination, indent=2, allow_nan=False)
        destination.write("\n")
