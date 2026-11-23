"""Run existing API and pattern checks, then test local schema references."""

import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator

subprocess.run(
    [sys.executable, "-B", str(Path(__file__).with_name("pattern_probe.py")), *sys.argv[1:]],
    check=True,
)
address, model = sys.argv[1:3]
output = Path(sys.argv[sys.argv.index("--out") + 1]).with_name("schema-references.json")
nested = {
    "$id": "urn:riley:root",
    "$defs": {"value": {"const": 1}},
    "type": "object",
    "properties": {
        "value": {
            "$id": "urn:riley:child",
            "$defs": {"value": {"const": 2}},
            "$ref": "#/$defs/value",
        },
    },
    "required": ["value"],
    "additionalProperties": False,
}
Draft202012Validator.check_schema(nested)
oracle = Draft202012Validator(nested)
assert not oracle.is_valid({"value": 1}) and oracle.is_valid({"value": 2})
crossing = {
    "$defs": {"child": {"$id": "urn:riley:child", "$defs": {"value": {"const": 2}}}},
    "$ref": "#/$defs/child/$defs/value",
}
local = {
    "$id": "urn:riley:root",
    "$defs": {"$id": {"const": 3}},
    "type": "object",
    "properties": {"$id": {"$ref": "#/$defs/$id"}},
    "required": ["$id"],
    "additionalProperties": False,
}
checks = []
for case, schema, error_fragment in (
    ("nested-id", nested, "embedded $id"),
    ("cross-resource-pointer", crossing, "$ref crossing an embedded $id"),
    ("root-id-local-pointer", local, None),
):
    for parameter in ("response_format", "tools"):
        payload = {
            "model": model, "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "user", "content": "Return the single object with key $id and value 3."}],
        }
        if parameter == "tools":
            payload["tools"] = [
                {"type": "function", "function": {"name": "capture", "parameters": schema}},
            ]
            payload["tool_choice"] = {"type": "function", "function": {"name": "capture"}}
        else:
            payload["response_format"] = {
                "type": "json_schema", "json_schema": {"name": "result", "schema": schema},
            }
        request = Request(
            f"http://{address}/v1/chat/completions", json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                status, body = response.status, json.load(response)
        except HTTPError as error:
            status, body = error.code, json.load(error)
        result = {"case": case, "parameter": parameter, "status": status}
        if error_fragment is not None:
            error = body.get("error", {})
            assert (status, error.get("code"), error.get("param")) == (400, "invalid_schema", parameter), body
            assert error_fragment in error["message"], body
            result["error"] = error
        else:
            assert status == 200, body
            choice, = body["choices"]
            if parameter == "tools":
                assert choice["finish_reason"] == "tool_calls", body
                call, = choice["message"]["tool_calls"]
                assert call["function"]["name"] == "capture", body
                raw = call["function"]["arguments"]
            else:
                assert choice["finish_reason"] == "stop", body
                raw = choice["message"]["content"]
            result["output"] = json.loads(raw)
            assert result["output"] == {"$id": 3} and Draft202012Validator(local).is_valid(result["output"])
        checks.append(result)
output.write_text(json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n")
