"""Replay lossless whole registers with both known and unknown source IDs."""

import argparse
import json
import random
import time
from pathlib import Path

from agent_tool_table import encode_record_table
from agent_workflow import fingerprint, stream
from long_tool_records import fixture


def obj(fields):
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


def case(kind, seed, count):
    req, expected = fixture(seed, count)
    records = json.loads(req["messages"][-1]["content"])
    entry = obj({"source": {"type": "string"}, "approved": {"type": "integer"}})
    if kind == "lookup":
        schema = obj({k: {"type": "integer"} for k in expected})
    elif kind == "scan":
        tags = ["suspended", "restored"]
        indexes = random.Random(seed + 44411).sample(range(count), 2)
        for tag, i in zip(tags, indexes, strict=True):
            records[i]["service"] = tag
        expected = {
            tag: {k: records[i][k] for k in ("source", "approved")}
            for tag, i in zip(tags, indexes, strict=True)
        }
        schema = obj({tag: entry for tag in tags})
        req["messages"][0]["content"] = (
            "Read the entire returned register. Find the one entry whose service is suspended and "
            "the one whose service is restored. Return only a JSON object with keys suspended and "
            "restored, each containing source (the exact source ID) and approved (its integer amount)."
        )
    elif kind == "max":
        top = max(r["approved"] for r in records)
        (winner,) = [r for r in records if r["approved"] == top]
        expected = {k: winner[k] for k in ("source", "approved")}
        schema = entry
        req["messages"][0]["content"] = (
            "Read every record. Find the single largest approved amount across the entire register. "
            "Return only a JSON object with source (the winning source ID) and approved (its integer amount)."
        )
    else:
        raise ValueError(kind)
    if kind != "lookup":
        req["messages"][0]["content"] += (
            " A table with columns and rows uses the listed column order."
        )
    req["max_tokens"] = 128 if kind != "lookup" else 96
    return req, records, expected, schema


def body(req, records, schema, table, strict):
    data = encode_record_table(records) if table else records
    req["messages"][-1]["content"] = json.dumps(data, separators=(",", ":"))
    if strict:
        req["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "register_result",
                "strict": True,
                "schema": schema,
            },
        }
    return req


def parse(text):
    def unique(pairs):
        value = dict(pairs)
        if len(value) != len(pairs):
            raise ValueError("duplicate JSON key")
        return value

    try:
        return json.loads(
            text.strip().removeprefix("```json").removesuffix("```").strip(),
            object_pairs_hook=unique,
        )
    except ValueError:
        return None


def run(address, out, reverse):
    config = json.loads(
        Path(__file__).with_name("full-register-registration.json").read_bytes()
    )
    rows = []
    for kind, specs in config["cases"].items():
        for seed, count in specs:
            cells = [(False, False), (False, True), (True, False), (True, True)]
            if reverse:
                cells.reverse()
            for table, strict in cells:
                req, records, expected, schema = case(kind, seed, count)
                start = time.monotonic_ns()
                req = body(req, records, schema, table, strict)
                m = {}
                r = {
                    "kind": kind,
                    "seed": seed,
                    "count": count,
                    "table": table,
                    "strict": strict,
                    "start_ns": start,
                    "result_hash": fingerprint(records),
                    "expected": expected,
                    "measurement": m,
                }
                try:
                    answer, calls = stream(address, req, m, timeout_s=600)
                    r.update(answer=answer, calls=calls)
                except (ValueError, OSError) as exc:
                    r["error"] = str(exc)
                rows.append(r)
                out.write_text(json.dumps(rows) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("out", type=Path)
    p.add_argument("--reverse", action="store_true")
    a = p.parse_args()
    run(a.address, a.out, a.reverse)
