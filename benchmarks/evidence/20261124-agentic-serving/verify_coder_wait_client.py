"""Compare the public wait-policy client on matched dependent interactions from two GPUs."""

import argparse
import hashlib
import json
from pathlib import Path

from agent_workflow import answer_format, fingerprint, fixture
from verify_coder_wait import check_preview
from verify_model_decode import natural, read


def score(record, seed, expected_decision, delay, has_reader):
    work = record["target"]
    x = natural(work, seed, 96, 200, delay=delay)
    assert x["verified_tool_calls"] == 2
    final = work["turns"][-1]
    assert final.get("status") == 200 and final["done_ns"]
    assert record["decision"] == expected_decision, (
        seed,
        record["decision"],
        expected_decision,
    )
    prepare = record["prefix"]
    check_preview(work, "boundary" if expected_decision == "submit" else "off", prepare)
    if expected_decision == "submit":
        assert (
            prepare["status"] == 200 and prepare["done_ns"] and "error" not in prepare
        )
        assert (
            work["turns"][1]["tool_start_ns"]
            <= prepare["sent_ns"]
            < work["turns"][1]["tool_ready_ns"]
        )
    ready = work["turns"][1]["tool_ready_ns"]
    assert ready <= final["sent_ns"]
    reader = None
    if has_reader:
        peer = record["peer"]
        pc = natural(peer, seed + 1, 2, 200, delay=0)
        assert pc["verified_tool_calls"] == 2
        events = peer["turns"][-1]["content_ns"]
        assert events[0] < ready < events[-1]
        reader = {
            "valid": pc["valid"],
            "gap_p95_ms": peer["final_gap_p95_ms"],
            "completion_ms": pc["last_final_ms"],
            "answer_hash": fingerprint(peer["turns"][-1]["answer"]),
        }
    return {
        "seed": seed,
        "decision": expected_decision,
        "valid": x["valid"],
        "cached": (final["usage"].get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        ),
        "prompt_tokens": final["usage"]["prompt_tokens"],
        "tokens": x["final_tokens"],
        "ttft_ms": x["final_submission_to_first_ms"],
        "complete_ms": x["last_final_ms"],
        "workflow_first_ms": x["first_final_ms"],
        "gap_p95_ms": work.get("final_gap_p95_ms"),
        "answer_hash": fingerprint(final["answer"]),
        "answer_format": answer_format(final["answer"], fixture(seed, 96)[2]),
        "preview_done_before_ready_ms": (
            (ready - prepare["done_ns"]) / 1e6 if prepare else None
        ),
        "reader": reader,
    }


def verify(root):
    assert (root / "exits.log").read_text().split() == ["0", "0"]
    for line in (root / "inputs.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    c = read(root / "coder-wait-client-registration.json")
    result = {}
    for gpu, order in c["order"].items():
        d = root / gpu
        assert read(d / "lease-after.json")["phase"] == "idle"
        assert (d / "binary.sha256").read_text().split()[
            0
        ] == "13b589df5c19a33c685ca1388398905f46a6cc566cbd1bed68f06b798080ca9f"
        shutdown = read(d / "shutdown.json")
        assert all(
            shutdown[k] == 0
            for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
        )
        assert not any(shutdown["allocation"].values())
        rows = read(d / "wait-policy.json")
        assert len(rows) == len(c["scenarios"]) * len(order)
        for pos, (group, enabled) in enumerate(order):
            for case, (label, off, actual, hint, reader) in enumerate(c["scenarios"]):
                r = rows[pos * len(c["scenarios"]) + case]
                assert r["group"] == group
                rec = r["result"]
                assert (rec["case"], rec["enabled"], rec["estimated_wait_ms"]) == (
                    label,
                    enabled,
                    hint,
                )
                decision = "control_off"
                if enabled:
                    decision = {
                        "repeat": "recent_prefix",
                        "short-hint": "short_or_unknown_wait",
                    }.get(label, "submit")
                cell = result.setdefault(group, {}).setdefault(label, {})
                cell["enabled" if enabled else "off"] = score(
                    rec, c["seeds"][group] + off, decision, actual, reader
                )
    gate = True
    for sub in result.values():
        for case, cell in sub.items():
            a, b = cell["off"], cell["enabled"]
            assert a["prompt_tokens"] == b["prompt_tokens"]
            cell["same_final_text"] = a["answer_hash"] == b["answer_hash"]
            cell["completion_change_ms"] = b["complete_ms"] - a["complete_ms"]
            cell["cached_change"] = b["cached"] - a["cached"]
            gate &= a["valid"] and b["valid"]
            if case in ("long-wait", "overlap"):
                gate &= cell["completion_change_ms"] < 0
            if a["reader"]:
                cell["reader_gap_ratio"] = (
                    b["reader"]["gap_p95_ms"] / a["reader"]["gap_p95_ms"]
                )
                gate &= (
                    a["reader"]["valid"]
                    and b["reader"]["valid"]
                    and cell["reader_gap_ratio"] <= 1.05
                )
    return {"cases": result, "registered_criteria_passed": gate}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    print(json.dumps(verify(p.parse_args().directory), indent=2))
