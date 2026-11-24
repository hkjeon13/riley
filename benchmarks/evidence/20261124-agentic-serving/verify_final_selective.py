"""Measure scoped final-turn catalog retention against the same physical reference."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from agent_workflow import fingerprint
from audit_final_layout_answers import PREFIX
from verify_final_layout import GROUPS, HERE, NAMES, check, inspect, read, tokenize


def verify(root, crossover, prior_raw, exe, tokenizer):
    reg = read(root / "final-layout-selective-registration.json")
    check(
        reg == read(HERE / "final-layout-selective-registration.json"),
        "unregistered trial",
    )
    for h, n in (
        line.split() for line in (root / "inputs.sha256").read_text().splitlines()
    ):
        check(sha256((root / n).read_bytes()).hexdigest() == h, f"changed {n}")
    baselines = read(crossover)["cells"]
    rows = {}
    comparisons = {}
    for gpu, order in reg["order"].items():
        check((root / f"exit-{gpu}").read_text().strip() == "0", "incomplete launch")
        prev = 0
        for policy in order:
            arm = f"gpu{gpu}-{policy}"
            path = root / arm
            refname = f"gpu{gpu}-riley-hide"
            refpath = prior_raw / refname
            lease = read(root / f"{arm}.lease.json")
            end = read(path / "shutdown.json")
            check(
                lease["phase"] == "idle"
                and lease["physical_index"] == int(gpu)
                and not lease["other_processes"]
                and lease["allocation_bytes"] == 20 * 2**30
                and lease["gpu"] in (path / "device.csv").read_text(),
                "device",
            )
            for f in ("binary.sha256", "server-arguments.txt", "device.csv"):
                check(
                    (path / f).read_bytes() == (refpath / f).read_bytes(),
                    "different config",
                )
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
                "shutdown",
            )
            groups = read(path / "workflows.json")
            check([(g["history"], g["seed"]) for g in groups] == list(GROUPS), "groups")
            norms = tokenize(groups, "riley", policy, exe, tokenizer)
            check(
                len(norms) == 60
                and all(n["matched"] and n["matched_length"] for n in norms.values()),
                "request identities",
            )
            spans = [w for g in groups for w in g["records"]]
            check(min(w["start_ns"] for w in spans) > prev, "run order")
            prev = max(w["end_ns"] for w in spans)
            for g in groups:
                group = f"{g['history']}-{g['seed']}"
                check(
                    [w["workflow_id"] for w in g["records"]] == list(NAMES), "records"
                )
                a = baselines[group][refname]
                b = {w["workflow_id"]: inspect(w, norms) for w in g["records"]}
                rows.setdefault(group, {})[arm] = b
                other = [n for n in NAMES if n != "long-tool-result"]
                check(
                    all(
                        b[n]["served_prompt_digest"] == a[n]["served_prompt_digest"]
                        for n in other
                    ),
                    "scope expanded beyond long tool results",
                )
                w = "long-tool-result"
                peer = "reader"
                check(
                    all(
                        b[n]["fixture_sha256"] == a[n]["fixture_sha256"] for n in NAMES
                    ),
                    "task mismatch",
                )
                pairs = {
                    k: [a[w].get(k), b[w].get(k)]
                    for k in (
                        "cached",
                        "completion",
                        "final_ttft_ms",
                        "first_final_ms",
                        "last_final_ms",
                    )
                }
                comparisons.setdefault(group, {})[arm] = {
                    **pairs,
                    "previous_valid_lost": [
                        n for n in NAMES if a[n]["ok"] and not b[n]["ok"]
                    ],
                    "unexpected_final_tool": [
                        n for n in NAMES if b[n].get("unexpected_tool")
                    ],
                    "unchanged_short_answer_count": sum(
                        b[n].get("generated_digest") == a[n].get("generated_digest")
                        for n in other
                    ),
                    "actual_overlap": b[peer]["first_ns"]
                    <= b[w]["sent_ns"]
                    < b[peer]["last_ns"],
                    "reader_last_ratio": b[peer]["last_final_ms"]
                    / a[peer]["last_final_ms"],
                    "reader_gap_ratio": b[peer]["gap_p95_ms"] / a[peer]["gap_p95_ms"],
                }
    reviewed = read(HERE / "final-selective-reviewed-answers.json")
    answers = []
    for group, arms in rows.items():
        if not group.startswith("200-"):
            continue
        for arm, cells in arms.items():
            w = cells["long-tool-result"]
            text = w["answer"]
            digest = fingerprint(text)
            header = PREFIX.match(text.replace("*", "").strip())
            grade = reviewed.get(digest)
            if grade:
                check(grade["unsupported_excerpt"] in text, "wrong quality observation")
            answers.append(
                {
                    "group": group,
                    "arm": arm,
                    "answer_sha256": digest,
                    "explanation_words": len(
                        text.replace("*", "").strip()[header.end() :].split()
                    )
                    if header
                    else None,
                    "unsupported_claim_witness": grade,
                }
            )
    return {
        "reviewed_long_history_answers": answers,
        "requests": 240,
        "submitted_workflows": 80,
        "valid_workflows": sum(
            t["ok"] for g in rows.values() for c in g.values() for t in c.values()
        ),
        "comparisons": comparisons,
        "cells": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "crossover", "prior_raw", "exe", "tokenizer"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            verify(a.root, a.crossover, a.prior_raw, a.exe, a.tokenizer), indent=2
        )
    )
