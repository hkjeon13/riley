"""Exercise request normalization and serving with identical tool schemas and histories."""

import argparse
import concurrent.futures
import json
import platform
import random
import subprocess
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from statistics import median
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    stream,
    turn_request,
)


def catalog(count, choice="auto"):
    tools = []
    for i in range(count):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"inventory_{i:03d}",
                    "description": f"Read warehouse {i} current bin count and last audit state.",
                    "parameters": {
                        "type": "object",
                        "properties": {"bin": {"type": "string", "enum": ["A3", "B1"]}},
                        "required": ["bin"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    q = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": "Use inventory_000 with bin A3."
                if choice == "named"
                else "Answer with the word OK.",
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        q["tools"] = tools
        q["tool_choice"] = (
            {"type": "function", "function": {"name": "inventory_000"}}
            if choice == "named"
            else choice
        )
    return q


def cases():
    out = {
        f"{n}-{choice}": catalog(n, choice)
        for n in (4, 32, 96)
        for choice in ("auto", "named", "none")
    }
    out["no-tools"] = catalog(0)
    extra = catalog(4, "none")
    extra["chat_template_kwargs"] = {
        "tools": extra["tools"],
        "messages": "shadow",
        "add_generation_prompt": False,
    }
    out["extra-template-tools"] = extra
    long = catalog(4, "none")
    long["messages"] = [
        {"role": "user", "content": "Summarize the audit status in two words."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "inventory_000", "arguments": '{"bin":"A3"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "The inventory audit is complete. " * 640,
        },
    ]
    out["long-result"] = long
    return out


def recorded(file):
    for group in json.loads(file.read_bytes()):
        for r in group["records"]:
            tools, prompt, docs = fixture(r["seed"], r["document_pairs"])
            msgs = history_messages(r["seed"], r["history_entries"]) + [
                {"role": "user", "content": prompt}
            ]
            for i, t in enumerate(r["turns"]):
                q = turn_request(
                    "m",
                    deepcopy(msgs),
                    tools,
                    ("list_reports", "read_reports", None)[i],
                )
                q["return_token_ids"] = True
                if fingerprint(q) != t["input_sha256"]:
                    raise ValueError("original request identity")
                yield f"recorded-turn-{i}", q, t["prompt_ids_sha256"]
                if "call" in t:
                    append_tool_result(msgs, [t["call"]], docs, i)


def cpu(root, model, file):
    items = list(recorded(file))
    synthetic = cases()
    for label, q in synthetic.items():
        for _ in range(12):
            items.append((label, q, None))
    result = {"host": platform.platform(), "recorded": 60, "samples": [], "probes": {}}
    for arm in ("control", "candidate"):
        result["probes"][arm] = sha256((root / f"probe-{arm}").read_bytes()).hexdigest()
    for pair in range(4):
        order = ("control", "candidate") if pair in (0, 3) else ("candidate", "control")
        index = list(range(len(items)))
        random.Random(56038 + pair).shuffle(index)
        payload = (
            "\n".join(json.dumps(items[i][1]) for i in [*index[:3], *index]) + "\n"
        )
        identities = {}
        for arm in order:
            run = subprocess.run(
                [str((root / f"probe-{arm}").resolve()), str(model.resolve())],
                input=payload,
                text=True,
                capture_output=True,
                check=True,
            )
            rows = [json.loads(line) for line in run.stdout.splitlines()][3:]
            samples = []
            for i, row in zip(index, rows, strict=True):
                label, q, old = items[i]
                digest = fingerprint(row["ids"])
                if old and digest != old:
                    raise ValueError("different GPU and CPU prompt")
                if identities.setdefault(i, row["ids"]) != row["ids"]:
                    raise ValueError("different token IDs")
                samples.append(
                    {
                        "case": label,
                        "index": i,
                        "input_sha256": fingerprint(q),
                        "tokens": len(row["ids"]),
                        "ids_sha256": digest,
                        "normalize_us": row["normalize_ns"] / 1000,
                        "encode_us": row["encode_ns"] / 1000,
                    }
                )
            result["samples"].append(
                {"pair": pair, "arm": arm, "observations": samples}
            )
    result["by_case"] = {}
    for label in sorted({row[0] for row in items}):
        med = {}
        for arm in ("control", "candidate"):
            groups = [
                [r for r in run["observations"] if r["case"] == label]
                for run in result["samples"]
                if run["arm"] == arm
            ]
            med[arm] = {
                "normalize_us": [
                    median(r["normalize_us"] for r in group) for group in groups
                ],
                "encode_us": [
                    median(r["encode_us"] for r in group) for group in groups
                ],
            }
        result["by_case"][label] = med
    return result


def serving(address, out):
    data = {}
    chosen = cases()
    for label in (
        "no-tools",
        "4-auto",
        "32-auto",
        "32-named",
        "32-none",
        "long-result",
    ):
        req = chosen[label]

        def call(_, req=req):
            m = {}
            answer, calls = stream(address, req, m)
            return {"measurement": m, "answer": answer, "calls": calls}

        data[label] = {"cold": call(0), "warm_serial": [call(i) for i in range(5)]}
        with concurrent.futures.ThreadPoolExecutor(8) as executor:
            data[label]["burst"] = list(executor.map(call, range(8)))
        out.write_text(json.dumps(data) + "\n")
    errors = {}
    invalid = catalog(32, "named")
    invalid["tools"][-1] = {"type": "function", "function": {}}
    unknown = catalog(4, "named")
    unknown["tool_choice"]["function"]["name"] = "other"
    for label, body in (("bad-unselected", invalid), ("unknown-name", unknown)):
        req = Request(
            f"http://{address}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=30):
                raise ValueError("accepted invalid catalog")
        except HTTPError as error:
            if error.code != 400:
                raise
            errors[label] = json.load(error)
    data["invalid"] = errors
    out.write_text(json.dumps(data) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    probe = sub.add_parser("cpu")
    for n in ("root", "model", "file", "out"):
        probe.add_argument(n, type=Path)
    live = sub.add_parser("serve")
    live.add_argument("address")
    live.add_argument("out", type=Path)
    a = p.parse_args()
    if a.mode == "cpu":
        a.out.write_text(json.dumps(cpu(a.root, a.model, a.file)) + "\n")
    else:
        serving(a.address, a.out)
