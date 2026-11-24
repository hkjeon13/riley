"""Run declared relational operations against full tool JSON before final generation."""

import argparse
import json
import time
from pathlib import Path

from agent_tool_aggregate import project_unique_integer_max
from agent_tool_projection import project_exact_records
from agent_workflow import fingerprint, stream
from full_register import case

HERE = Path(__file__).resolve().parent


def reduce(records, kind):
    if kind == "scan":
        return project_exact_records(
            records, key="service", requested_ids=("suspended", "restored")
        )
    if kind == "max":
        return project_unique_integer_max(records, key="approved")
    raise ValueError("unknown query operation")


def run(address, out, gpu):
    reg = json.loads((HERE / "register-reduction-registration.json").read_text())
    rows = []
    visit = 0
    for kind, specs in reg["cases"].items():
        for seed, count in specs:
            policies = (
                ("full", "reduced") if (gpu + visit) % 2 == 0 else ("reduced", "full")
            )
            visit += 1
            for policy in policies:
                req, records, expected, _ = case(kind, seed, count)
                original = json.dumps(records, separators=(",", ":"))
                start = time.monotonic_ns()
                if policy == "reduced":
                    parsed = json.loads(original)
                    retained = reduce(parsed, kind)
                    content = json.dumps(retained, separators=(",", ":"))
                else:
                    content = original
                req["messages"][-1]["content"] = content
                prepared = time.monotonic_ns()
                m = {}
                row = {
                    "kind": kind,
                    "seed": seed,
                    "count": count,
                    "policy": policy,
                    "expected": expected,
                    "start_ns": start,
                    "prepared_ns": prepared,
                    "original_hash": fingerprint(original),
                    "sent_content_hash": fingerprint(content),
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
