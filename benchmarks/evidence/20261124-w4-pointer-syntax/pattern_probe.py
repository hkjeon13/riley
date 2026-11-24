"""Run the ordinary API suite followed by focused character-class probes."""

import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import check_distribution_gpu as checker

subprocess.run([sys.executable, "-B", str(checker.API_CHECK), *sys.argv[1:]], check=True)
address, model = sys.argv[1:3]
out = Path(sys.argv[sys.argv.index("--out") + 1]).with_name("pattern-classes.json")
results = []
for name, pattern, expected_error in [
    ("quote-range", "^[!-#]$", None),
    ("backslash-range", "^[A-z]$", None),
    ("unsupported-class-escape", r"^[\n]$", r"pattern class escape \n"),
    ("unsupported-unicode-escape", r"^[\u0061]$", r"pattern class escape \u"),
    ("unsupported-negated-space", r"^[^\s]$", r"pattern \s inside a negated class"),
]:
    schema = {"type": "object", "properties": {"value": {"type": "string", "pattern": pattern}},
              "required": ["value"], "additionalProperties": False}
    for tool in (False, True):
        payload = {"model": model, "temperature": 0, "max_tokens": 64,
                   "messages": [{"role": "user", "content": 'Set value to a double quote (") when allowed, otherwise a backslash.'}]}
        if tool:
            payload["tools"] = [{"type": "function", "function": {"name": "capture", "parameters": schema}}]
            payload["tool_choice"] = {"type": "function", "function": {"name": "capture"}}
        else:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "character", "schema": schema}}
        request = Request(f"http://{address}/v1/chat/completions", json.dumps(payload).encode(),
                          {"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=30) as response:
                status, reply = response.status, json.load(response)
        except HTTPError as error:
            status, reply = error.code, json.load(error)
        parameter = "tools" if tool else "response_format"
        if expected_error is not None:
            observed = reply.get("error", {})
            assert (status, observed.get("code"), observed.get("param")) == (400, "invalid_schema", parameter), reply
            assert expected_error in observed["message"], reply
            results.append({"case": name, "parameter": parameter, "status": status, "code": observed["code"]})
            continue
        assert status == 200, reply
        choice = reply["choices"][0]
        if tool:
            assert choice["finish_reason"] == "tool_calls", reply
            call, = choice["message"]["tool_calls"]
            assert call["function"]["name"] == "capture", reply
            text = call["function"]["arguments"]
        else:
            assert choice["finish_reason"] == "stop", reply
            text = choice["message"]["content"]
        decoded = json.loads(text)
        assert list(decoded) == ["value"] and re.fullmatch(pattern, decoded["value"]), reply
        results.append({"case": name, "parameter": parameter, "status": status, "output": decoded,
                        "finish_reason": choice["finish_reason"]})
out.write_text(json.dumps({"status": "passed", "checks": results}, indent=2) + "\n")
