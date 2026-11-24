"""Bind generated source IDs to the records returned by the typed tool operation."""

import argparse
import json
import time
from pathlib import Path

from agent_workflow import fingerprint, stream
from full_register import case, obj
from register_reduction import reduce


def result_schema(kind, records):
    allowed = sorted({row["source"] for row in records})
    if not allowed or not all(isinstance(s, str) and s for s in allowed):
        raise ValueError("source IDs must be nonempty strings")
    entry = obj(
        {"source": {"type": "string", "enum": allowed}, "approved": {"type": "integer"}}
    )
    if kind == "scan":
        return obj({state: entry for state in ("suspended", "restored")})
    if kind == "max":
        return entry
    raise ValueError("unknown registered operation")


def run(address, out, gpu):
    reg = json.loads(
        Path(__file__).with_name("register-provenance-registration.json").read_bytes()
    )
    rows = []
    case_no = 0
    for kind, specs in reg["cases"].items():
        for seed, count in specs:
            first_strict = bool((gpu + case_no) % 2)
            case_no += 1
            for strict in (first_strict, not first_strict):
                req, records, expected, _ = case(kind, seed, count)
                full = json.dumps(records, separators=(",", ":"))
                start = time.monotonic_ns()
                retained = reduce(json.loads(full), kind)
                text = json.dumps(retained, separators=(",", ":"))
                req["messages"][-1]["content"] = text
                if strict:
                    schema = result_schema(kind, retained)
                    req["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "tool_result",
                            "strict": True,
                            "schema": schema,
                        },
                    }
                m = {}
                row = {
                    "kind": kind,
                    "seed": seed,
                    "count": count,
                    "strict": strict,
                    "start_ns": start,
                    "full_hash": fingerprint(full),
                    "wire": fingerprint(req),
                    "expected": expected,
                    "measurement": m,
                }
                try:
                    answer, calls = stream(address, req, m, timeout_s=600)
                    row.update(answer=answer, calls=calls)
                except (ValueError, OSError) as exc:
                    row["error"] = str(exc)
                rows.append(row)
                out.write_text(json.dumps(rows) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("out", type=Path)
    p.add_argument("gpu", type=int, choices=(6, 7))
    a = p.parse_args()
    run(a.address, a.out, a.gpu)
