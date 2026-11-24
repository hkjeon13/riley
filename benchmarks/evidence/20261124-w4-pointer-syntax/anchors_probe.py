"""Exercise JSON Schema branch anchors through packaged chat and tools."""
import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator


def probe(address, model, output):
    cases = (
        ("start-branch", "^cat|dog$", "catfish"),
        ("end-branch", "^cat|dog$", "sheepdog"),
        ("unanchored-middle-branch", "^cat$|dog|^eel$", "hotdogs"),
        ("escaped-dollar-and-class", r"^(cat|dog)$|\$[|]", "a$|b"),
        ("unsupported-interior-anchor", "ca$t|dog", None),
    )
    checks = []
    for name, pattern, expected in cases:
        schema = {
            "type": "object", "required": ["value"], "additionalProperties": False,
            "properties": {"value": {"type": "string", "pattern": pattern}},
        }
        message = f"Set the value field to the exact string {expected!r}. Copy every character."
        for parameter in ("response_format", "tools"):
            payload = {
                "model": model, "temperature": 0, "max_tokens": 64,
                "messages": [{"role": "user", "content": message}],
            }
            if parameter == "tools":
                payload["tools"] = [{"type": "function", "function": {"name": "capture", "parameters": schema}}]
                payload["tool_choice"] = {"type": "function", "function": {"name": "capture"}}
            else:
                payload["response_format"] = {
                    "type": "json_schema", "json_schema": {"name": "result", "schema": schema},
                }
            request = Request(f"http://{address}/v1/chat/completions", json.dumps(payload).encode(),
                              {"Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=30) as response:
                    status, body = response.status, json.load(response)
            except HTTPError as error:
                status, body = error.code, json.load(error)
            result = {"case": name, "pattern": pattern, "parameter": parameter, "status": status, "expected": expected}
            if expected is None:
                error = body.get("error", {})
                if (status, error.get("code"), error.get("param")) != (400, "invalid_schema", parameter):
                    raise ValueError((result, body))
                if "pattern operator $" not in error.get("message", ""):
                    raise ValueError((result, body))
                result["error"] = error
            else:
                if status != 200:
                    raise ValueError((result, body))
                choice, = body["choices"]
                if parameter == "tools":
                    call, = choice["message"]["tool_calls"]
                    if choice["finish_reason"] != "tool_calls" or call["function"]["name"] != "capture":
                        raise ValueError((result, body))
                    value = json.loads(call["function"]["arguments"])
                else:
                    if choice["finish_reason"] != "stop":
                        raise ValueError((result, body))
                    value = json.loads(choice["message"]["content"])
                Draft202012Validator(schema).validate(value)
                if value != {"value": expected}:
                    raise ValueError((result, body))
                result["output"] = value
            checks.append(result)
    output.write_text(json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n")


if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-B", str(Path(__file__).with_name("reference_probe.py")), *sys.argv[1:]],
        check=True,
    )
    result_path = Path(sys.argv[sys.argv.index("--out") + 1]).with_name("pattern-anchors.json")
    probe(*sys.argv[1:3], result_path)
