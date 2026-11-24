"""Verify selected whole records against full tool output and time to final content."""

import argparse
import json
from pathlib import Path

from agent_workflow import fingerprint
from long_tool_records import extract, fixture
from verify_long_tile import read, receipt


def compare(root):
    results = {}
    configs = []
    gpus = set()
    all_pass = True
    for gpu, order in (("gpu6", ("full", "lookup")), ("gpu7", ("lookup", "full"))):
        args, device, binary = receipt(root / gpu, "page-v1")
        configs.append((args, binary))
        gpus.add(device)
        if "--prefix-cache-blocks" in args:
            raise ValueError("prefix reuse was not disabled")
        rows = read(root / gpu / "projection.json")
        cases = read(root / "projection-registration.json")["cohort"]
        if [(r["seed"], r["policy"]) for r in rows] != [
            (seed, k) for seed, _ in cases for k in order
        ]:
            raise ValueError("missing or misordered arms")
        pairs = []
        for i, (seed, n) in enumerate(cases):
            pair = {"seed": seed, "records": n}
            for r in rows[2 * i : 2 * i + 2]:
                policy = r["policy"]
                req, expected = fixture(seed, n)
                content = req["messages"][-1]
                if (
                    r["full_tool_result_hash"] != fingerprint(content)
                    or expected != r["expected"]
                ):
                    raise ValueError("wrong full tool result or ground truth")
                original = json.loads(content["content"])
                if policy == "lookup":
                    selected = [v for v in original if v["source"] in expected]
                    if len(selected) != 2:
                        raise ValueError("ambiguous source fixture")
                    content["content"] = json.dumps(selected, separators=(",", ":"))
                m = r["measurement"]
                if (
                    r["records_sent"] != (2 if policy == "lookup" else n)
                    or set(r["ids_from_request"]) != set(expected)
                    or fingerprint(req) != r["request_hash"]
                    or r["request_hash"] != m["input_sha256"]
                ):
                    raise ValueError("wire did not receive the exact whole records")
                events = m["content_ns"]
                usage = m["usage"]
                ok = (
                    extract(r.get("answer", "")) == expected
                    and bool(m.get("done_ns"))
                    and m["status"] == 200
                )
                if (
                    ok != r["valid"]
                    or usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    != 0
                ):
                    raise ValueError("quality disagreement or cache reuse")
                pair[policy] = {
                    "correct": ok,
                    "prompt_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["completion_tokens"],
                    "prepare_ms": (m["sent_ns"] - r["start_ns"]) / 1e6,
                    "first_ms": (events[0] - r["start_ns"]) / 1e6,
                    "last_ms": (events[-1] - r["start_ns"]) / 1e6,
                }
            pair["passed"] = pair["lookup"]["correct"] and all(
                pair["lookup"][f] < pair["full"][f] for f in ("first_ms", "last_ms")
            )
            all_pass &= pair["passed"]
            pairs.append(pair)
        results[gpu] = pairs
    if len(gpus) != 2 or configs[0] != configs[1]:
        raise ValueError("replica settings differ or duplicate physical device")
    return {
        "binary_sha256": configs[0][1],
        "both_released": True,
        "all_gates": all_pass,
        "cases": results,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(compare(p.parse_args().root), indent=2))
