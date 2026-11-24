"""Verify tool eligibility, native prefix tokens and user-visible final performance."""

import argparse
import json
import re
import subprocess
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    percentile,
    turn_request,
)

HERE = Path(__file__).resolve().parent
NAMES = ("cold", "immediate", "delayed", "reader", "long-tool-result")
GROUPS = ((0, 761), (200, 761), (200, 18081), (200, 19959))


def read(path):
    return json.loads(path.read_bytes())


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def requests(record, engine, layout, named=False):
    tools, prompt, docs = fixture(record["seed"], record["document_pairs"])
    check(fingerprint([prompt, docs]) == record["fixture_sha256"], "different fixture")
    messages = history_messages(record["seed"], record["history_entries"])
    messages.append({"role": "user", "content": prompt})
    for index, turn in enumerate(record["turns"]):
        req = turn_request(
            "m", messages, tools, ("list_reports", "read_reports", None)[index]
        )
        req["return_token_ids"] = True
        keep = layout == "keep"
        if index == 2 and layout in ("selective", "defined"):
            check(messages[-1]["role"] == "tool", "no completed tool result")
            original = messages[-1]["content"]
            check(json.loads(original) == docs, "return differs from original reports")
            keep = len(original.encode("utf-8")) >= 4096
            if keep and layout == "defined":
                from final_report_presentation import P95_EVIDENCE

                record_json = {"metric_definition": P95_EVIDENCE, "reports": docs}
                req["messages"] = [
                    *messages[:-1],
                    {**messages[-1], "content": json.dumps(record_json)},
                ]
        if index == 2 and keep and engine == "riley":
            if named:
                req["include_tools_when_tool_choice_none"] = True
            else:
                req["chat_template_kwargs"] = {"tools": tools}
        check(
            fingerprint(req) == turn.get("input_sha256"),
            "submitted canonical body mismatch",
        )
        if index == 2 and keep:
            req.pop("include_tools_when_tool_choice_none", None)
            req["chat_template_kwargs"] = {"tools": tools}
        yield json.dumps(req)
        if "call" in turn:
            append_tool_result(messages, [turn["call"]], docs, index)


def tokenize(groups, engine, layout, executable, tokenizer, named=False):
    bodies = []
    turns = []
    for group in groups:
        for record in group["records"]:
            bodies.extend(requests(record, engine, layout, named))
            turns.extend(record["turns"])
    command = [str(executable.resolve()), str(tokenizer.resolve())]
    p = subprocess.run(
        command,
        input="\n".join(bodies) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    rows = {}
    for turn, line in zip(turns, p.stdout.splitlines(), strict=True):
        ids = json.loads(line)["ids"]
        digest = fingerprint(ids)
        # A renderer discrepancy is evidence, not an invitation to silently
        # use Riley's tokenization for a different engine.
        remote_digest = turn.get("prompt_ids_sha256")
        rows[id(turn)] = {
            "ids": ids,
            "offline_digest": digest,
            "served_digest": remote_digest,
            "matched": remote_digest == digest,
            "matched_length": len(ids) == turn.get("usage", {}).get("prompt_tokens"),
        }
    return rows


def shared(a, b):
    return next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )


def inspect(record, normalized):
    out = {
        "ok": record["ok"],
        "error": record.get("error"),
        "fixture_sha256": record["fixture_sha256"],
        "turns": len(record["turns"]),
        "workflow": record["workflow_id"],
        "start_ns": record["start_ns"],
        "end_ns": record["end_ns"],
    }
    if len(record["turns"]) != 3 or not record["turns"][-1].get("content_ns"):
        return out
    second, final = record["turns"][1:]
    result = normalized[id(final)]
    prefix = (
        min(
            shared(normalized[id(second)]["ids"], result["ids"]), len(result["ids"]) - 1
        )
        // 16
        * 16
    )
    text = final.get("answer", "")
    markers = bool(
        re.search(r"</?tool_call>|\[TOOL_CALLS\]|<\|python_tag\|>", text, re.IGNORECASE)
    )
    forbidden = (
        "last_call_fragment_ns" in final
        or final.get("finish") == "tool_calls"
        or markers
    )
    events = final["content_ns"]
    usage = final["usage"]
    fp = final.get("generated_ids", [])
    gaps = [(b - a) / 1e6 for a, b in pairwise(events)]
    out.update(
        status=final["status"],
        finish=final.get("finish"),
        done=bool(final.get("done_ns")),
        cached=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        prompt=usage["prompt_tokens"],
        completion=usage["completion_tokens"],
        generated_digest=fingerprint(fp),
        generated_id_count=len(fp),
        served_prompt_digest=result["served_digest"],
        native_match=result["matched"],
        native_length_match=result["matched_length"],
        native_prior_turn_prefix=prefix,
        unexpected_tool=forbidden,
        answer=text,
        first_ns=events[0],
        last_ns=events[-1],
        sent_ns=final["sent_ns"],
        words=len(text.split()),
        first_final_ms=(events[0] - record["start_ns"]) / 1e6,
        last_final_ms=(events[-1] - record["start_ns"]) / 1e6,
        final_ttft_ms=(events[0] - final["sent_ns"]) / 1e6,
        gap_p95_ms=percentile(gaps, 0.95),
        final_tpot_ms=(events[-1] - events[0]) / 1e6 / max(1, len(fp) - 1)
        if len(fp) == usage["completion_tokens"]
        else None,
    )
    return out


def analyze(root, executable, tokenizer):
    reg = read(root / "final-layout-registration.json")
    check(reg == read(HERE / "final-layout-registration.json"), "changed registration")
    for line in (root / "inputs.sha256").read_text().splitlines():
        h, f = line.split()
        check(sha256((root / f).read_bytes()).hexdigest() == h, f)
    all_cells = {}
    launches = {}
    signatures = {}
    for gpu, order in reg["orders"].items():
        check((root / f"exit-{gpu}").read_text().strip() == "0", "unfinished crossover")
        previous = 0
        for arm in order:
            engine, layout = arm.split("-")
            name = f"gpu{gpu}-{arm}"
            path = root / name
            lease = read(root / f"{name}.lease.json")
            check(
                lease["phase"] == "idle"
                and not lease["other_processes"]
                and lease["physical_index"] == int(gpu)
                and lease["allocation_bytes"] == 20 * 2**30,
                "GPU lease",
            )
            if engine == "riley":
                end = read(path / "shutdown.json")
                check(
                    not any(
                        end[k]
                        for k in (
                            "active_requests",
                            "waiting_requests",
                            "kv_allocated_blocks",
                        )
                    )
                    and not any(end["allocation"].values()),
                    "server release",
                )
                check(lease["gpu"] in (path / "device.csv").read_text(), "served GPU")
                check(
                    "fixed-tile-unsplit-v1" in (path / "server.log").read_text(),
                    "attention mode",
                )
            else:
                cmd = read(path / "command.json")
                check(
                    ("--exclude-tools-when-tool-choice-none" in cmd)
                    == (layout == "hide"),
                    "vLLM tool layout",
                )
            if engine == "riley":
                signature = (
                    (path / "binary.sha256").read_text().split()[0],
                    (path / "server-arguments.txt").read_text(),
                )
            else:
                signature = (
                    (path / "version.txt").read_text().strip(),
                    tuple(
                        item
                        for i, item in enumerate(cmd)
                        if item != "--exclude-tools-when-tool-choice-none"
                        and (i == 0 or cmd[i - 1] != "--port")
                    ),
                )
            check(
                signatures.setdefault(engine, signature) == signature,
                f"changed engine configuration in {name}",
            )
            data = read(path / "workflows.json")
            check(
                [(d["history"], d["seed"]) for d in data] == list(GROUPS),
                "workload inventory",
            )
            tokenized = tokenize(data, engine, layout, executable, tokenizer)
            launches[name] = {
                "turns": len(tokenized),
                "native_ids_equal": sum(t["matched"] for t in tokenized.values()),
                "returned_prompt_ids": sum(
                    bool(t["served_digest"]) for t in tokenized.values()
                ),
                "native_length_equal": sum(
                    t["matched_length"] for t in tokenized.values()
                ),
            }
            for g in data:
                check(
                    [r["workflow_id"] for r in g["records"]] == list(NAMES),
                    "missing request",
                )
                key = f"{g['history']}-{g['seed']}"
                cell = {r["workflow_id"]: inspect(r, tokenized) for r in g["records"]}
                all_cells.setdefault(key, {})[name] = cell
            spans = [r for g in data for r in g["records"]]
            check(min(r["start_ns"] for r in spans) > previous, "wrong run order")
            previous = max(r["end_ns"] for r in spans)
    comparisons = {}
    for key, cells in all_cells.items():
        comparisons[key] = {}
        for gpu in reg["orders"]:
            for engine in ("riley", "vllm"):
                a, b = [
                    cells[f"gpu{gpu}-{engine}-{policy}"] for policy in ("hide", "keep")
                ]
                for n in NAMES:
                    check(
                        a[n]["fixture_sha256"] == b[n]["fixture_sha256"], "changed task"
                    )
                lost = [n for n in NAMES if a[n]["ok"] and not b[n]["ok"]]
                writer = "long-tool-result"
                failed = [n for n in NAMES if not b[n]["ok"]]
                summary = {
                    "previously_valid_lost": lost,
                    "candidate_failed": failed,
                    "candidate_tool_violations": [
                        n for n in NAMES if b[n].get("unexpected_tool")
                    ],
                }
                if all(x[n].get("done") for x in (a, b) for n in ("reader", writer)):
                    for k in (
                        "cached",
                        "native_prior_turn_prefix",
                        "prompt",
                        "completion",
                        "first_final_ms",
                        "last_final_ms",
                        "final_ttft_ms",
                        "final_tpot_ms",
                    ):
                        summary[k] = [x[writer][k] for x in (a, b)]
                    for k in ("gap_p95_ms", "last_final_ms"):
                        summary["reader_" + k + "_ratio"] = (
                            b["reader"][k] / a["reader"][k]
                        )
                    summary["overlap"] = [
                        x["reader"]["first_ns"]
                        <= x[writer]["sent_ns"]
                        < x["reader"]["last_ns"]
                        for x in (a, b)
                    ]
                comparisons[key][f"gpu{gpu}-{engine}"] = summary
    engine_comparison = {}
    for group, arms in all_cells.items():
        engine_comparison[group] = {}
        for gpu in reg["orders"]:
            for layout in ("hide", "keep"):
                records = [
                    arms[f"gpu{gpu}-{e}-{layout}"]["long-tool-result"]
                    for e in ("riley", "vllm")
                ]
                if not all(r.get("done") for r in records):
                    continue
                a, b = records
                engine_comparison[group][f"gpu{gpu}-{layout}"] = {
                    "prompt_ids_equal": bool(a["served_prompt_digest"])
                    and a["served_prompt_digest"] == b["served_prompt_digest"],
                    "generated_ids_equal": a["generated_digest"]
                    == b["generated_digest"],
                    "completion_tokens": [a["completion"], b["completion"]],
                    "riley_minus_vllm_last_ms": a["last_final_ms"] - b["last_final_ms"],
                    "riley_minus_vllm_final_ttft_ms": a["final_ttft_ms"]
                    - b["final_ttft_ms"],
                    "riley_minus_vllm_prior_ms": (
                        (a["sent_ns"] - a["start_ns"]) - (b["sent_ns"] - b["start_ns"])
                    )
                    / 1e6,
                    "riley_minus_vllm_stream_ms": (
                        (a["last_ns"] - a["first_ns"]) - (b["last_ns"] - b["first_ns"])
                    )
                    / 1e6,
                }
    return {
        "launches": launches,
        "comparisons": comparisons,
        "same_layout_engine_comparison": engine_comparison,
        "submitted_workflows": sum(
            len(v) for group in all_cells.values() for v in group.values()
        ),
        "fully_valid_workflows": sum(
            w["ok"]
            for group in all_cells.values()
            for v in group.values()
            for w in v.values()
        ),
        "http_turns": sum(r["turns"] for r in launches.values()),
        "cells": all_cells,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "executable", "tokenizer"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    print(json.dumps(analyze(a.root, a.executable, a.tokenizer), indent=2))
