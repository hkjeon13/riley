"""Replay exact-output tool contracts from original serialized data, including fallbacks."""

import argparse
import json
import platform
import sys
import time
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "live"))
from agent_tool_result import exact_tool_result
from agent_workflow import fingerprint, percentile
from full_register import case

FIELDS = {"source": str, "approved": int}


def adapt(records, kind):
    if kind == "scan":
        return exact_tool_result(
            records,
            operation="unique_matches",
            key="service",
            fields=FIELDS,
            requested_values=["suspended", "restored"],
        )
    return exact_tool_result(
        records, operation="unique_integer_max", key="approved", fields=FIELDS
    )


def run():
    original = json.loads(
        (HERE / "register-provenance-registration.json").read_bytes()
    )["cases"]
    held = json.loads((HERE / "exact-result-registration.json").read_bytes())[
        "new_cases"
    ]
    out = {}
    for kind in original:
        for seed, count in original[kind] + held[kind]:
            _, records, answer, _ = case(kind, seed, count)
            source = json.dumps(records, separators=(",", ":")).encode()
            if kind == "scan":
                truth = {}
                for tag in ("suspended", "restored"):
                    (row,) = [r for r in records if r["service"] == tag]
                    truth[tag] = {"source": row["source"], "approved": row["approved"]}
            else:
                ranking = sorted(
                    ((r["approved"], r["source"]) for r in records), reverse=True
                )
                if ranking[0][0] == ranking[1][0]:
                    raise ValueError("unexpected tie in fixture")
                truth = {"source": ranking[0][1], "approved": ranking[0][0]}
            if truth != answer:
                raise ValueError("independent oracle mismatch")
            samples = []
            outputs = set()
            for _ in range(25):
                start = time.perf_counter_ns()
                response = json.dumps(
                    adapt(json.loads(source), kind), separators=(",", ":")
                ).encode()
                samples.append((time.perf_counter_ns() - start) / 1e6)
                outputs.add(response)
            if len(outputs) != 1 or json.loads(response) != truth:
                raise ValueError("record-to-value binding lost")
            out[f"{kind}-{seed}"] = {
                "records": count,
                "input_bytes": len(source),
                "output_bytes": len(response),
                "output_hash": fingerprint(json.loads(response)),
                "new": [seed, count] in held[kind],
                "cpu_samples_ms": samples,
                "cpu_ms_p50": percentile(samples, 0.5),
                "cpu_ms_p95": percentile(samples, 0.95),
            }
    valid = [
        {"source": "X", "service": "suspended", "approved": 100},
        {"source": "Y", "service": "restored", "approved": 10},
    ]
    fallback = [
        (valid + valid[:1], "scan"),
        (valid[:1], "scan"),
        ([{"source": 9, "approved": 5}], "max"),
        ([{"source": "X", "approved": True}], "max"),
        ([{"source": "X", "approved": 5}, {"source": "Z", "approved": 5}], "max"),
        ([{"approved": 4}], "max"),
        ([], "max"),
    ]
    if any(adapt(records, k) is not None for records, k in fallback):
        raise ValueError("invalid inputs need the caller's fallback")
    if (
        exact_tool_result(valid, operation="summarize", key="approved", fields=FIELDS)
        is not None
    ):
        raise ValueError("undeclared free-form query produced a result")
    return {
        "environment": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "repeats": 25,
        "source_sha256": {
            name: sha256((HERE.parents[1] / "live" / name).read_bytes()).hexdigest()
            for name in (
                "agent_tool_result.py",
                "agent_tool_aggregate.py",
                "agent_tool_projection.py",
            )
        },
        "cases": out,
        "fallback_cases_passed": len(fallback) + 1,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    print(json.dumps(run(), indent=2))
