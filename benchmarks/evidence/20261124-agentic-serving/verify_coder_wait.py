"""Reconstruct dependent inputs, verify actual preview boundaries and compare user-visible latencies."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from agent_workflow import (
    answer_format,
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    percentile,
    turn_request,
)
from join_trace import join
from verify_model_decode import check_frozen_requests, natural, read


def check_preview(work, mode, preview):
    if mode == "off":
        assert not preview
        return
    tools, prompt, docs = fixture(work["seed"], work["document_pairs"])
    messages = history_messages(work["seed"], work["history_entries"]) + [
        {"role": "user", "content": prompt}
    ]
    append_tool_result(messages, [work["turns"][0]["call"]], docs, 0)
    if mode == "boundary":
        call = work["turns"][1]["call"]
        messages.extend(
            [
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": ""},
            ]
        )
    prepared = turn_request("m", messages, tools, None)
    prepared["max_tokens"] = 1
    assert fingerprint(prepared) == preview["input_sha256"]


def summarize(raw, mode, spec, log, ident):
    seed, depth, notes, reader_seed = spec
    work, peer, preview = raw["target"], raw["reader"], raw["preview"]
    metrics = natural(work, seed, notes, depth)
    assert metrics["verified_tool_calls"] == 2
    completed = work["turns"][-1]
    assert completed["turn"] == 2 and completed["input_sha256"] == ident["input"]
    ready = work["turns"][1]["tool_ready_ns"]
    assert ready <= completed["sent_ns"] and completed["done_ns"]
    check_preview(work, mode, preview)
    first, last = completed["content_ns"][0], completed["content_ns"][-1]
    times = join(log, [work], details=True)
    final = times[-1]
    usage = completed["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    assert final["prefill_rows"] + cached == usage["prompt_tokens"]
    fmt = answer_format(completed["answer"], fixture(seed, notes)[2])
    intervals = [b - a for a, b in itertools.pairwise(completed["content_ns"])]
    cost = {
        key: sum(p[key] for p in raw["after"].values())
        - sum(p[key] for p in raw["before"].values())
        for key in ("prefill_tokens", "decode_tokens", "iterations", "gpu_execution_ns")
    }
    # Recompute the bucket-independent workload delta using complete member sets.
    ids = {
        int(row["response_id"].removeprefix("cmpl-"), 16)
        for w in (work, peer)
        if w
        for row in w["turns"]
    }
    if "response_id" in preview:
        ids.add(int(preview["response_id"].removeprefix("cmpl-"), 16))
    observed = Counter()
    seen = set()
    for line in log.splitlines():
        if not line.startswith("AGENT_STEP "):
            continue
        _, _, _, ns, raw_members = line.split(maxsplit=4)
        members = json.loads(raw_members)
        members_ids = {m[0] for m in members}
        if ids & members_ids:
            assert members_ids <= ids
            seen.update(members_ids)
            observed["gpu_execution_ns"] += int(ns)
            observed["iterations"] += 1
            for _, kind, rows, *_ in members:
                observed[{"p": "prefill_tokens", "d": "decode_tokens"}[kind]] += rows
    assert seen == ids and observed == cost
    reader = None
    if reader_seed is not None:
        check = natural(peer, reader_seed, 2, 0, delay=0)
        pe = peer["turns"][-1]["content_ns"]
        assert pe[0] < ready < pe[-1]
        reader = {
            "valid": check["valid"],
            "gap_p95_ms": percentile(
                [(b - a) / 1e6 for a, b in itertools.pairwise(pe)], 0.95
            ),
            "completion_ms": (pe[-1] - peer["start_ns"]) / 1e6,
            "tokens": check["final_tokens"],
            "answer_sha256": fingerprint(peer["turns"][-1]["answer"]),
        }
    elif peer is not None:
        raise ValueError("unexpected peer")
    return {
        "seed": seed,
        "valid": metrics["valid"],
        "prompt_hash": ident["tokens"],
        "prompt_tokens": usage["prompt_tokens"],
        "cached": cached,
        "new_rows": final["prefill_rows"],
        "prefill_chunk_rows": [c["rows"] for c in final["chunks"]],
        "answer_format": fmt,
        "prefill_gpu_ms": final["prefill_participating_gpu_ms"],
        "preview_ok": (
            bool(
                preview.get("status") == 200
                and preview.get("done_ns")
                and not preview.get("error")
            )
            if preview
            else None
        ),
        "preview_error": preview.get("error"),
        "preview_cached": (
            (preview.get("usage", {}).get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0
            )
            if preview
            else None
        ),
        "preview_prompt": preview.get("usage", {}).get("prompt_tokens"),
        "preview_done_before_ready_ms": (
            (ready - preview["done_ns"]) / 1e6 if "done_ns" in preview else None
        ),
        "ready_to_first_ms": (first - ready) / 1e6,
        "ttft_ms": metrics["final_submission_to_first_ms"],
        "workflow_first_ms": (first - work["start_ns"]) / 1e6,
        "workflow_last_ms": (last - work["start_ns"]) / 1e6,
        "final_tokens": metrics["final_tokens"],
        "answer_sha256": fingerprint(completed["answer"]),
        "final_gap_p95_ms": percentile(intervals, 0.95) / 1e6,
        "reader": reader,
        "cost": cost,
    }


def verify(root):
    assert (root / "exits.log").read_text().split() == ["0", "0"]
    for line in (root / "inputs.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    for line in (root / "token-audit.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    token_audit = read(root / "tokens.json")
    conf = read(root / "coder-wait-registration.json")
    cases = [(*c, None) for c in conf["cases"]] + [
        (s, 100, 96, r) for s, r in conf["held_out_readers"]
    ]
    arms = {}
    comparisons = {}
    binary = (root / "trace-server.sha256").read_text().split()[0]
    cli = None
    for gpu, modes in conf["orders"].items():
        arms[gpu] = {}
        for mode in modes:
            d = root / f"gpu{gpu}" / mode
            assert (d / "binary.sha256").read_text().split()[0] == binary
            assert read(d / "lease-after.json")["phase"] == "idle"
            arguments = (d / "server-arguments.txt").read_text()
            if cli is None:
                cli = arguments
            assert cli == arguments
            shutdown = read(d / "shutdown.json")
            assert all(
                shutdown[k] == 0
                for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
            )
            assert all(v == 0 for v in shutdown["allocation"].values())
            check_frozen_requests(root, d)
            ids = read(d / "identity.json")
            log = (d / "server.log").read_text()
            assert "RILEY_NATIVE_ATTENTION mode=fixed-tile-unsplit-v1" in log
            arms[gpu][mode] = {
                c[0]: summarize(raw, mode, c, log, ids[str(c[0])])
                for raw, c in zip(read(d / "cases.json"), cases, strict=True)
            }
        comparisons[gpu] = {}
        for mode in ("history", "boundary"):
            gates = Counter()
            denied = []
            deltas = {}
            for seed, b in arms[gpu][mode].items():
                a = arms[gpu]["off"][seed]
                assert (a["prompt_hash"], a["prompt_tokens"]) == (
                    b["prompt_hash"],
                    b["prompt_tokens"],
                )
                bound = token_audit[str(seed)][mode]
                assert b["preview_prompt"] == bound["tokens"]
                deltas[seed] = {
                    "cached_change": b["cached"] - a["cached"],
                    "shared_complete_tokens": bound["shared_complete_tokens"],
                    "ttft_change_ms": b["ttft_ms"] - a["ttft_ms"],
                    "workflow_first_change_ms": b["workflow_first_ms"]
                    - a["workflow_first_ms"],
                    "workflow_last_change_ms": b["workflow_last_ms"]
                    - a["workflow_last_ms"],
                    "gpu_time_ratio": b["cost"]["gpu_execution_ns"]
                    / a["cost"]["gpu_execution_ns"],
                    "same_final_text": b["answer_sha256"] == a["answer_sha256"],
                }
                checks = {
                    "preview_success": b["preview_ok"],
                    "basic_validity": not a["valid"] or b["valid"],
                    "completion_guard": b["workflow_last_ms"]
                    <= 1.05 * a["workflow_last_ms"],
                    "gpu_budget": b["cost"]["gpu_execution_ns"]
                    <= 1.05 * a["cost"]["gpu_execution_ns"],
                }
                if seed in (833, 861, 902):
                    checks["high_opportunity_both_faster"] = (
                        b["ttft_ms"] < a["ttft_ms"]
                        and b["workflow_last_ms"] < a["workflow_last_ms"]
                    )
                if a["reader"]:
                    checks["reader_gap_guard"] = (
                        b["reader"]["gap_p95_ms"] <= 1.05 * a["reader"]["gap_p95_ms"]
                    )
                    checks["reader_validity"] = (
                        not a["reader"]["valid"] or b["reader"]["valid"]
                    )
                for k, v in checks.items():
                    if v:
                        gates[k] += 1
                    else:
                        denied.append([seed, k])
            comparisons[gpu][mode] = {
                "passed": not denied,
                "passes": dict(gates),
                "changes_from_off": deltas,
                "failed": denied,
            }
    return {"arms": arms, "candidate_gates": comparisons, "binary_sha256": binary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
