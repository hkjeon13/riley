#!/usr/bin/env python3
"""Check long-context chat schemas and forced tools against an independent oracle."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator

from long_accuracy import CONTEXT, read_workload, token_ids


INPUT_LENGTHS = (255, 1024, 7836, 7917)
LIMIT = 64
OVER_LIMIT = 96
SCHEMA = {"type": "object", "properties": {"mode": {"type": "string", "enum": ["ready", 1]},
          "count": {"type": "integer", "const": 7}},
          "required": ["mode", "count"], "additionalProperties": False}
TOOL = "record_status"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON member: {key}")
        result[key] = value
    return result


def decode_reply(raw, streamed):
    if not streamed:
        body = json.loads(raw)
        choice, = body["choices"]
        return choice["finish_reason"], choice["prompt_token_ids"], choice["token_ids"], body["usage"], choice["message"]

    events = [line.removeprefix("data: ") for line in raw.splitlines() if line.startswith("data: ")]
    require(events and events[-1] == "[DONE]" and events.count("[DONE]") == 1, "missing or duplicate SSE terminator")
    chunks = [json.loads(line) for line in events[:-1]]
    require(all("error" not in chunk and len(chunk["choices"]) <= 1 for chunk in chunks), "unexpected SSE error or choices")
    choices = [choice for chunk in chunks for choice in chunk["choices"]]
    finish, = [choice["finish_reason"] for choice in choices if choice["finish_reason"] is not None]
    require(choices[-1]["finish_reason"] == finish, "SSE data after terminal choice")
    prompt, = [choice["prompt_token_ids"] for choice in choices if "prompt_token_ids" in choice]
    usage, = [chunk["usage"] for chunk in chunks if chunk.get("usage") is not None]
    require(not chunks[-1]["choices"] and chunks[-1]["usage"] == usage, "missing final SSE usage event")
    tokens = [token for choice in choices for token in choice.get("token_ids", [])]
    parts = [choice["delta"] for choice in choices]
    message = {"content": "".join(part.get("content") or "" for part in parts),
               "tool_calls": [call for part in parts for call in part.get("tool_calls", [])]}
    return finish, prompt, tokens, usage, message


def chat_request(address, model, case, kind, streamed, limit):
    instruction = "Use record_status with mode ready and count 7." if kind == "tool" else "Return JSON with mode ready and count 7."
    body = {"model": model, "temperature": 0, "max_tokens": limit, "return_token_ids": True, "stream": streamed,
            "messages": [{"role": "user", "content": case.prompt + "\n\nIgnore the previous task. " + instruction}]}
    if streamed:
        body["stream_options"] = {"include_usage": True}
    if kind == "tool":
        body["tools"] = [{"type": "function", "function": {"name": TOOL, "parameters": SCHEMA}}]
        body["tool_choice"] = {"type": "function", "function": {"name": TOOL}}
    else:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "status", "schema": SCHEMA}}

    return Request(f"http://{address}/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode(),
                   {"Content-Type": "application/json"})


def request(address, model, case, kind, streamed):
    call = chat_request(address, model, case, kind, streamed, LIMIT)
    try:
        with urlopen(call, timeout=180) as response:
            require(response.status == 200, "chat HTTP request failed")
            raw = response.read().decode()
    except HTTPError as error:
        with error:
            detail = error.read(4096).decode(errors="replace")
        raise ValueError(f"{key(case, kind, streamed)}: HTTP {error.code}: {detail}") from error
    finish, prompt, tokens, usage, message = decode_reply(raw, streamed)
    prompt, tokens = token_ids(prompt), token_ids(tokens)
    require(len(case.expected_ids) <= len(prompt) <= CONTEXT - LIMIT and 0 < len(tokens) <= LIMIT,
            f"{case.label} rendered outside the supported token range")
    require(usage["prompt_tokens"] == len(prompt)
            and usage["completion_tokens"] in (len(tokens), len(tokens) + 1)
            and usage["total_tokens"] == len(prompt) + usage["completion_tokens"], "inconsistent chat token usage")
    content = message["content"] or ""
    call_name = None
    value = content
    if kind == "tool":
        tool, = message["tool_calls"]
        function = tool["function"]
        require(finish == "tool_calls" and tool["type"] == "function" and tool["id"]
                and function["name"] == TOOL and not content, "forced tool call contract failed")
        call_name, value = function["name"], function["arguments"]
    else:
        require(finish == "stop" and not message.get("tool_calls"), "schema response did not finish successfully")
    Draft202012Validator(SCHEMA).validate(json.loads(value, object_pairs_hook=unique_object))
    return {"prompt": prompt, "tokens": tokens, "usage": usage, "finish": finish,
            "content": content, "tool_name": call_name, "value": value}


def run_cell(address, model, cases, concurrency):
    peak = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = [pool.submit(request, address, model, *case) for case in cases]
        while not all(future.done() for future in pending):
            with urlopen(f"http://{address}/metrics", timeout=5) as response:
                peak = max(peak, json.load(response)["active_requests"])
            time.sleep(0.02)
        results = [future.result() for future in pending]
    require((1 if concurrency == 1 else 2) <= peak <= concurrency,
            f"C{concurrency} did not observe the expected request overlap (peak {peak})")
    return results, peak


def key(case, kind, streamed):
    return f"{case.label}/{kind}/{('sse' if streamed else 'json')}"


def check_context_rejection(address, model, case, prompt_tokens):
    requested = prompt_tokens + OVER_LIMIT
    require(requested > CONTEXT, "tool fixture does not exceed context at the larger output limit")
    try:
        with urlopen(chat_request(address, model, case, "tool", False, OVER_LIMIT), timeout=30):
            raise ValueError("the over-context tool request was accepted")
    except HTTPError as error:
        with error:
            detail = json.load(error)["error"]
        require(error.code == 400 and detail["code"] == "invalid_request", "unexpected over-context error")
        return {"case": case.label, "sequence_tokens_requested": requested, "max_sequence_tokens": CONTEXT,
                "max_output_tokens": OVER_LIMIT, "http_status": error.code, "error_code": detail["code"]}


def check(address, model, path):
    workload = read_workload(path)
    cases = [(case, kind, streamed) for case in workload.cases if len(case.expected_ids) in INPUT_LENGTHS
             for kind in ("schema", "tool") for streamed in (False, True)]
    reference, peak = run_cell(address, model, cases, 1)
    expected = {key(*case): result for case, result in zip(cases, reference, strict=True)}
    for case, kind, stream in cases:
        if stream:
            require(expected[key(case, kind, True)] == expected[key(case, kind, False)],
                    f"{case.label}/{kind}: C1 JSON and SSE responses differ")
    longest = max(workload.cases, key=lambda case: len(case.expected_ids))
    rejection = check_context_rejection(address, model, longest, len(expected[key(longest, "tool", False)]["prompt"]))
    modes = {"c1": {"requests": len(reference), "observed_peak_active": peak}}
    short = [case for case in cases if len(case[0].expected_ids) <= 1024]
    for concurrency, scheduled in ((8, cases), (32, [*cases, *(short * 2)])):
        results, peak = run_cell(address, model, scheduled, concurrency)
        for case, result in zip(scheduled, results, strict=True):
            require(result == expected[key(*case)], f"C{concurrency}: output or usage differs for {key(*case)}")
        modes[f"c{concurrency}"] = {"requests": len(results), "exact_matches": len(results), "observed_peak_active": peak}
    return {"status": "passed", "workload_sha256": workload.sha256, "jsonschema_version": version("jsonschema"),
            "modes": modes, "c1_transport_pairs": len(cases) // 2, "context_rejection": rejection,
            "cases": {name: {"usage": result["usage"], "finish": result["finish"], "value": result["value"],
                             "response_sha256": sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()}
                      for name, result in expected.items()}}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("workload", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = check(args.address, args.model, args.workload)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
