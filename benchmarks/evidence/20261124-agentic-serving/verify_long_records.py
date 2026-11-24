"""Rebuild paired tool-return requests and check their source-to-value bindings."""

import argparse
import json
from pathlib import Path

from agent_workflow import fingerprint
from long_tool_records import extract, fixture
from verify_long_tile import read, receipt


def compare(root):
    a = receipt(root / "page", "page-v1")
    if a != receipt(root / "tile", "fixed-tile-unsplit-v1"):
        raise ValueError("configuration, GPU or binary differs")
    cases = read(root / "tool-records-registration.json")["cases"]
    rows = {k: read(root / k / "tool-records.json") for k in ("page", "tile")}
    cases_out = []
    for (seed, count), page, tile in zip(
        cases, rows["page"], rows["tile"], strict=True
    ):
        req, expected = fixture(seed, count)
        pair = {"seed": seed, "records": count, "expected": expected}
        for key, row in (("page", page), ("tile", tile)):
            m = row["measurement"]
            if (
                row["expected"] != expected
                or row["request_sha256"] != fingerprint(req)
                or m["input_sha256"] != row["request_sha256"]
                or row["seed"] != seed
                or row["records"] != count
            ):
                raise ValueError("fixture or wire input mismatch")
            events = m.get("content_ns", [])
            correct = (
                extract(row.get("answer", "")) == expected
                and bool(m.get("done_ns"))
                and m.get("status") == 200
            )
            if correct != row["valid"]:
                raise ValueError("validator disagreement")
            usage = m.get("usage") or {}
            pair[key] = {
                "valid": correct,
                "answer": row.get("answer"),
                "error": row.get("error") or m.get("stream_error"),
                "input": usage.get("prompt_tokens"),
                "output": usage.get("completion_tokens"),
                "cache": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                "first_ms": (events[0] - m["sent_ns"]) / 1e6 if events else None,
                "last_ms": (events[-1] - m["sent_ns"]) / 1e6 if events else None,
            }
        pair["same_input_and_cache"] = all(
            pair["page"][f] == pair["tile"][f] for f in ("input", "cache")
        )
        cases_out.append(pair)
    return {
        "physical_gpu": a[1],
        "binary_sha256": a[2],
        "leases_idle_and_shutdown_zero": True,
        "cases": cases_out,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(compare(p.parse_args().root), indent=2))
