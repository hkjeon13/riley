"""Validate live Riley responses against a separately implemented schema oracle."""
import http.client
from importlib.metadata import version
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

port = int(sys.argv[1])
out = Path(sys.argv[2])
assert version("jsonschema") == "4.23.0"
defaults = {"model": "m", "temperature": 0, "max_tokens": 80,
            "messages": [{"role": "user", "content": "Return a status record with mode ready."}]}
good = {"type": "object", "properties": {"mode": {"type": "string", "enum": ["ready", 1]}},
        "required": ["mode"], "additionalProperties": False}
overlap = {"oneOf": [{"type": "integer"}, {"type": "integer", "minimum": 0}]}
results = []


def format_of(schema):
    return {"type": "json_schema", "json_schema": {"name": "status", "schema": schema}}


def request(name, **kwargs):
    payload = {**defaults, **kwargs}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=90)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(payload, ensure_ascii=False),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read().decode()
        row = {"name": name, "request": payload, "status": response.status, "body": raw}
        results.append(row)
        return row
    finally:
        connection.close()


def choice(row):
    assert row["status"] == 200, row
    if row["request"].get("stream"):
        payloads = [line.removeprefix("data: ") for line in row["body"].splitlines() if line.startswith("data: ")]
        assert payloads[-1] == "[DONE]", row
        chunks = [json.loads(line) for line in payloads[:-1]]
        assert not any("error" in chunk for chunk in chunks), row
        picks = [ch for chunk in chunks for ch in chunk["choices"]]
        terminal = [ch["finish_reason"] for ch in picks if ch["finish_reason"] is not None]
        assert len(terminal) == 1, row
        return terminal[0], {"content": "".join(c["delta"].get("content", "") for c in picks),
                             "tool_calls": [call for c in picks for call in c["delta"].get("tool_calls", [])]}
    pick = json.loads(row["body"])["choices"][0]
    return pick["finish_reason"], pick["message"]


def error(row, code, parameter, mentions):
    actual = json.loads(row["body"])["error"]
    assert row["status"] == 400 and actual["code"] == code and actual["param"] == parameter, row
    assert mentions in actual["message"], row


def format_checks():
    validator = Draft202012Validator(good)
    for stream in (False, True):
        row = request(f"typed enum stream={stream}", response_format=format_of(good), stream=stream,
                      logprobs=True, top_logprobs=2)
        finish, result = choice(row)
        assert finish == "stop" and validator.is_valid(json.loads(result["content"])), row
        error(request(f"oneOf stream={stream}", response_format=format_of(overlap), stream=stream),
              "invalid_schema", "response_format", "oneOf")
    for constraint, keyword in (({"const": 1, "minimum": 2}, "minimum"),
                                ({"anyOf": [{"type": "integer"}, {"type": "string"}], "type": "integer"}, "type")):
        error(request(f"sibling {keyword}", response_format=format_of(constraint)),
              "invalid_schema", "response_format", keyword)
    error(request("grammar with explicit stop", response_format=format_of(good), stop="}"),
          "unsupported_parameter", "stop", "stop")
    partial = request("grammar reaches token limit", response_format=format_of(good), max_tokens=1, stream=True)
    reason, _ = choice(partial)
    assert reason == "length", partial


def tools_checks():
    tool = {"type": "function", "function": {"name": "set_status", "parameters": good}}
    force = {"type": "function", "function": {"name": "set_status"}}
    validator = Draft202012Validator(good)
    for mode, stream in (("required", True), (force, False)):
        row = request(f"constrained tool stream={stream}", tools=[tool], tool_choice=mode, stream=stream)
        finish, result = choice(row)
        calls = result["tool_calls"]
        assert finish == "tool_calls" and len(calls) == 1 and calls[0]["function"]["name"] == "set_status", row
        assert validator.is_valid(json.loads(calls[0]["function"]["arguments"])), row
    bad_tool = {"type": "function", "function": {"name": "set_status", "parameters": overlap}}
    error(request("forced tool rejects oneOf", tools=[bad_tool], tool_choice=force),
          "invalid_schema", "tools", "oneOf")


try:
    format_checks()
    tools_checks()
finally:
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(f"{len(results)} live structured output checks passed")
