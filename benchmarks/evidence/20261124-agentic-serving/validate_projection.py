"""Compare full records with a typed-key projection on identical final-answer tasks."""

import argparse
import json
import re
import time
from pathlib import Path

from agent_tool_projection import project_exact_records
from agent_workflow import fingerprint, stream
from long_tool_records import extract, fixture


def check_fallback():
    original = [{"source": "A", "amount": 4}, {"source": "B", "amount": 7}]
    for ids in (None, [], ["X"], ["A", "A"], "A", {"A": 1}, {"A"}, 3, iter(["A"])):
        assert (
            project_exact_records(original, key="source", requested_ids=ids) is original
        )
    for key in (None, ["source"], ""):
        assert project_exact_records(original, key=key, requested_ids=["A"]) is original
    rows = iter(original)
    assert project_exact_records(rows, key="source", requested_ids=["A"]) is rows
    assert list(rows) == original
    duplicate = original + [original[0]]
    assert (
        project_exact_records(duplicate, key="source", requested_ids=["A"]) is duplicate
    )
    malformed = original + [{"opaque": "record"}]
    assert (
        project_exact_records(malformed, key="source", requested_ids=["B"]) is malformed
    )
    result = project_exact_records(original, key="source", requested_ids=["B"])
    assert len(result) == 1 and result[0] is original[1]


def run(address, out, reverse):
    check_fallback()
    result = []
    for seed, size in ((541, 210), (733, 620), (862, 890)):
        for policy in ("lookup", "full") if reverse else ("full", "lookup"):
            start = time.monotonic_ns()
            req, expected = fixture(seed, size)
            user = req["messages"][0]["content"]
            ids = re.findall(r"\bF\d+-\d{4}\b", user)
            if len(ids) != 2:
                raise ValueError("the registered exact-key lookup must have two IDs")
            content = req["messages"][-1]
            full_hash = fingerprint(content)
            data = json.loads(content["content"])
            if policy == "lookup":
                data = project_exact_records(data, key="source", requested_ids=ids)
                content["content"] = json.dumps(data, separators=(",", ":"))
            measurement = {}
            row = {
                "seed": seed,
                "policy": policy,
                "full_tool_result_hash": full_hash,
                "ids_from_request": ids,
                "records_sent": len(data),
                "request_hash": fingerprint(req),
                "expected": expected,
            }
            try:
                answer, calls = stream(address, req, measurement, timeout_s=180)
                row.update(
                    answer=answer,
                    valid=not calls
                    and extract(answer) == expected
                    and bool(measurement.get("done_ns")),
                )
            except (OSError, ValueError) as exc:
                row.update(valid=False, error=str(exc))
            row["measurement"] = measurement
            row["start_ns"] = start
            result.append(row)
            out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("out", type=Path)
    p.add_argument("--reverse", action="store_true")
    a = p.parse_args()
    run(a.address, a.out, a.reverse)
