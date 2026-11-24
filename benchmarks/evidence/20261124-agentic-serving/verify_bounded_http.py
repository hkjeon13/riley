"""Re-encode received HTTP bodies, compare complete final generations, and audit catalog budgets."""

import argparse
import json
from collections import Counter
from pathlib import Path

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    turn_request,
)
from budgeted_tool_prefix import PromptProfiler, budgeted_preview
from first_tool_capacity import common
from first_tool_schedules import partials, render
from tool_wait_trial import prepare_prefix
from verify_coder_named_steps import capture_final
from verify_named_cpu_live import verify as check_generation


def read(p):
    return json.loads(p.read_text())


def checkpoint_run(root, control_root, profiler, tokenizer):
    current = check_generation(root, profiler, tokenizer)
    control = check_generation(control_root, profiler, tokenizer)
    assert control["engine"] == current["engine"]
    [obs] = read(root / "inference.json")
    [record] = read(root / "client.json")
    work = record["work"]
    reference = next(j for j in control["jobs"] if j["seed"] == work["seed"])
    old = read(control_root / "inference.json")[control["jobs"].index(reference)]
    assert (
        obs["output_ids"] == old["output_ids"]
        and obs["prompt_ids_sha256"] == old["prompt_ids_sha256"]
    )
    preview = read(root / "http-preview.json")
    [server] = preview["server"]
    client = preview["client"]
    plan = client["plan"]
    assert len(record["invocations"]) == 2 and len(record["receipts"]) == 1
    first, second, final = work["turns"]
    assert (
        first["tool_start_ns"]
        <= client["sent_ns"]
        <= client["done_ns"]
        <= final["sent_ns"]
    )
    assert second["tool_ready_ns"] <= final["sent_ns"] and not client.get(
        "cancel_requested_ns"
    )
    assert (
        client["status"] == 200
        and final["status"] == 200
        and client["response_id"] != final["response_id"]
    )
    partial = server["preview_request"]
    assert partial["tool_choice"] == "none" and partial["max_tokens"] == 1
    tool_texts = [m["content"] for m in partial["messages"] if m["role"] == "tool"]
    assert tool_texts in ([], [""])
    assert client["input_sha256"] == fingerprint(partial)
    tools, prompt, docs = fixture(work["seed"], work["document_pairs"])
    messages = history_messages(work["seed"], work["history_entries"]) + [
        {"role": "user", "content": prompt}
    ]
    known = prepare_prefix(
        turn_request("m", list(messages), tools, "list_reports"),
        first["call"],
        include_completed_call=True,
    )
    for i, t in enumerate((first, second)):
        append_tool_result(messages, [t["call"]], docs, i)
    complete = turn_request("m", messages, tools, None)
    assert fingerprint(complete) == final["input_sha256"]
    with PromptProfiler(profiler, tokenizer) as native:
        planned, receipt = budgeted_preview(known, plan["budget_tokens"], native.ids)
        assert receipt == plan and planned == partial
    pids, fids = render([partial, complete], profiler, tokenizer)
    assert fingerprint(pids) == plan["ids_sha256"] == server["preview_ids_sha256"]
    assert (
        client["usage"] == server["preview_usage"]
        and client["usage"]["prompt_tokens"]
        == len(pids)
        <= plan["budget_tokens"]
        == read(root / "named-cpu-registration.json")["http_budget"]
    )
    assert common(pids, fids) == server["common"]
    assert server["reused"] == min(common(pids, fids), len(fids) - 1) // 16 * 16 > 0
    assert (
        server["first_forward_tokens"] == len(fids) - server["reused"]
        and fingerprint(fids) == old["prompt_ids_sha256"]
    )
    return {
        "http_model_requests": 2,
        "tool_invocations": 2,
        "preview_budget": plan["budget_tokens"],
        "preview_tokens": len(pids),
        "shared_with_final": server["common"],
        "reused": server["reused"],
        "final_tokens": len(fids),
        "first_final_forward_tokens": server["first_forward_tokens"],
        "output_tokens_equal_to_control": len(obs["output_ids"]),
        "finish": obs["finish"],
        "both_sources": current["jobs"][0]["both_sources"],
        "three_values": current["jobs"][0]["all_three_values"],
        "preview_completed_before_final_ms": (final["sent_ns"] - client["done_ns"])
        / 1e6,
    }


def catalog_cases(root, profiler, tokenizer):
    cells = []
    with PromptProfiler(profiler, tokenizer) as native:
        for gpu in ("gpu6", "gpu7"):
            for wave in read(root / gpu / "selected/waves.json"):
                cap = 1024 // sum(j["role"] == "target" for j in wave["jobs"]) * 16
                native_cache = {}

                def encode(req, native_cache=native_cache):
                    key = fingerprint(req)
                    if key not in native_cache:
                        native_cache[key] = native.ids(req)
                    return native_cache[key]

                decisions = Counter()
                for j in wave["jobs"]:
                    starts = []

                    def start(i, req, row, starts=starts):
                        if i == 0:
                            starts.append(
                                prepare_prefix(
                                    req, row["call"], include_completed_call=True
                                )
                            )

                    final, _, _ = capture_final(
                        j["work"], j["role"] == "reader", on_tool_start=start
                    )
                    [known] = starts
                    chosen, receipt = budgeted_preview(known, cap, encode)
                    assert chosen is not None
                    original = (
                        known
                        if len(encode(known)) <= cap
                        else next(
                            s
                            for s in reversed(partials(known))
                            if len(encode(s)) <= cap
                        )
                    )
                    assert chosen == original
                    ids = encode(chosen)
                    assert len(ids) <= cap and common(ids, encode(final)) >= 16
                    decisions[receipt["decision"]] += 1
                cells.append(
                    {
                        "gpu": gpu,
                        "size": wave["size"],
                        "base": wave["base"],
                        "cap": cap,
                        "matched": sum(decisions.values()),
                        "decisions": dict(decisions),
                    }
                )
    return cells


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "control", "catalog", "profiler", "tokenizer", "out"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    report = {
        "cpu": checkpoint_run(a.root, a.control, a.profiler, a.tokenizer),
        "catalog": catalog_cases(a.catalog, a.profiler, a.tokenizer),
    }
    a.out.write_text(json.dumps(report, indent=2) + "\n")
