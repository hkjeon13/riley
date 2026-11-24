"""Compare exact token IDs or validation errors for supported and rejected chat forms."""

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

from agent_workflow import fingerprint
from chat_render_trial import catalog


def fixtures():
    base = catalog(4, "auto")
    named = catalog(4, "named")
    plain = catalog(0)
    yield (
        "unicode-parts",
        {
            **plain,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "보관함 Ω"},
                        {"type": "text", "text": "\n🚃"},
                    ],
                    "name": "reviewer",
                }
            ],
        },
    )
    yield (
        "kwargs-precedence",
        {
            **base,
            "chat_template_kwargs": {
                "tools": [{"broken": True}],
                "messages": [{}],
                "bos_token": "wrong",
                "eos_token": "wrong",
                "add_generation_prompt": False,
                "enable_thinking": False,
            },
        },
    )
    yield (
        "hidden-kwargs",
        {
            **base,
            "tool_choice": "none",
            "chat_template_kwargs": {"tools": base["tools"]},
        },
    )
    yield (
        "no-parameters",
        {
            **named,
            "tools": [{"type": "function", "function": {"name": "inventory_000"}}],
        },
    )
    yield (
        "required-parallel",
        {**base, "tool_choice": "required", "parallel_tool_calls": True},
    )
    params = {
        "type": "object",
        "properties": {"bin": {"$ref": "#/$defs/ID"}},
        "required": ["bin"],
        "additionalProperties": False,
        "$defs": {"ID": {"type": "string", "enum": ["A3", "B1"]}},
    }
    yield (
        "schema-local-ref",
        {
            **named,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "inventory_000", "parameters": params},
                }
            ],
        },
    )
    for label, arguments in (
        ("argument-object", {"bin": "A3"}),
        ("argument-string", '{"bin":"A3"}'),
        ("malformed-history-arguments", "not-json"),
    ):
        yield (
            label,
            {
                **base,
                "messages": [
                    {"role": "user", "content": "Read."},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c",
                                "type": "function",
                                "function": {
                                    "name": "inventory_000",
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "inventory_000",
                        "tool_call_id": "c",
                        "content": "7",
                    },
                ],
            },
        )
    yield "bad-hidden-tool", {**base, "tool_choice": "none", "tools": [{"type": "bad"}]}
    yield (
        "invalid-unselected-name",
        {**named, "tools": [*named["tools"], {"type": "function", "function": {}}]},
    )
    yield (
        "invalid-unselected-parameters",
        {
            **named,
            "tools": [
                *named["tools"],
                {
                    "type": "function",
                    "function": {"name": "ignored", "parameters": None},
                },
            ],
        },
    )
    yield "invalid-choice", {**base, "tool_choice": "unexpected"}
    yield (
        "invalid-content-part",
        {
            **plain,
            "messages": [
                {"role": "user", "content": [{"type": "image", "url": "none"}]}
            ],
        },
    )
    yield (
        "unsupported-message-attribute",
        {**plain, "messages": [{"role": "user", "content": "q", "unknown": True}]},
    )
    yield "forced-and-format", {**named, "response_format": {"type": "json_object"}}
    yield (
        "ignored-invalid-tools-empty-conversation",
        {**plain, "tool_choice": "none", "tools": [{"type": "bad"}], "messages": []},
    )


def run(root, models):
    result = {}
    for family, path, flags in models:
        data = {}
        for name, q in fixtures():
            outcomes = []
            for arm in ("control", "candidate"):
                call = subprocess.run(
                    [str((root / f"probe-{arm}").resolve()), str(path), *flags],
                    input=json.dumps(q) + "\n",
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if call.returncode:
                    if not call.stderr.startswith("Error:"):
                        raise ValueError(call.stderr)
                    outcomes.append({"invalid": call.stderr.strip()})
                else:
                    ids = json.loads(call.stdout)["ids"]
                    outcomes.append({"valid": ids})
            if outcomes[0] != outcomes[1]:
                raise ValueError(f"different {family}/{name}")
            outcome = outcomes[0]
            data[name] = {
                "input_sha256": fingerprint(q),
                **(
                    {
                        "tokens": len(outcome["valid"]),
                        "token_ids_sha256": fingerprint(outcome["valid"]),
                    }
                    if "valid" in outcome
                    else outcome
                ),
            }
        result[family] = {
            "files_sha256": {
                f: sha256((path / f).read_bytes()).hexdigest()
                for f in ("tokenizer.json", "tokenizer_config.json")
            },
            "outcomes": data,
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "qwen", "llama"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            run(a.root, [("qwen", a.qwen, []), ("llama", a.llama, ["--llama3"])]),
            indent=2,
        )
    )
