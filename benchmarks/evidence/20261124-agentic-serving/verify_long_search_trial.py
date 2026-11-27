"""Check the paired long-search results, latency, and GPU reclamation."""

import argparse
import json
import re
from pathlib import Path

from benchmarks.live.agent_workflow import fingerprint, percentile
from benchmarks.live.long_search_trial import SOURCE, corpus_cases, grade
from examples.long_search import retrieve, search_request


def load(path):
    return json.loads(path.read_text())


def normalize_citations(answer):
    return re.sub(rf"\[\s*({SOURCE})\s*\]", r"[\1]", answer, flags=re.IGNORECASE)


def gpu_processes(path):
    return {
        int(line.split(",", 1)[0])
        for line in path.read_text().splitlines()
        if "," in line
    }


def check_shutdown(root, mode, holder):
    directory = root / mode
    assert (directory / "exit.txt").read_text().strip() == "0", mode
    state = load(directory / "shutdown.json")
    fields = ("active_requests", "waiting_requests", "kv_allocated_blocks")
    assert not any(state[key] for key in fields), (mode, "request or KV leak")
    assert not any(state["allocation"].values()), (mode, "allocation leak")
    assert not any(state["counters"].values()), (
        mode,
        "request errors",
        state["counters"],
    )
    pid = int((directory / "server-pid.txt").read_text())
    observed = gpu_processes(directory / "compute-during.txt")
    assert observed == {holder, pid}, (mode, "unexpected GPU PIDs", observed)
    assert gpu_processes(directory / "compute-after.csv") == {holder}, mode
    return sum(
        line.isdecimal()
        for line in (directory / "compute-during.txt").read_text().splitlines()
    )


def summarize(rows):
    measurements = [row["measurement"] for row in rows]
    problems = [row["problems"] for row in rows]
    canonical = [
        grade(normalize_citations(row["answer"]), row["expected"]) for row in rows
    ]
    absent_ids = [
        set(row["expected"]) - set(re.findall(SOURCE, row["answer"].casefold()))
        for row in rows
    ]
    inputs = [m["usage"]["prompt_tokens"] for m in measurements]
    return {
        "attempted": len(rows),
        "stream_done": sum(bool(m.get("done_ns")) for m in measurements),
        "correct": sum(row["correct"] for row in rows),
        "correct_citation_whitespace_normalized": sum(
            not any(issue.values()) and m["finish"] == "stop"
            for issue, m in zip(canonical, measurements, strict=True)
        ),
        "matching_amounts": sum(not issue["missing_values"] for issue in problems),
        "all_sources_cited": sum(not issue["missing_sources"] for issue in problems),
        "omitted_source_queries": sum(
            bool(issue["missing_sources"]) for issue in problems
        ),
        "omitted_source_queries_whitespace_normalized": sum(
            bool(issue["missing_sources"]) for issue in canonical
        ),
        "source_id_absent_even_outside_citations_queries": sum(
            bool(ids) for ids in absent_ids
        ),
        "strictly_missing_citation_but_literal_ids_present_queries": sum(
            bool(issue["missing_sources"]) and not missing
            for issue, missing in zip(problems, absent_ids, strict=True)
        ),
        "omitted_source_count": sum(
            len(issue["missing_sources"]) for issue in problems
        ),
        "duplicate_source_queries": sum(
            bool(issue["duplicate_sources"]) for issue in problems
        ),
        "bad_binding_queries": sum(
            bool(
                issue["missing_pairs"]
                and not issue["missing_sources"]
                and not issue["missing_values"]
            )
            for issue in problems
        ),
        "all_relevant_sources_sent": sum(
            set(row["expected"]) == set(row["target_coverage"]) for row in rows
        ),
        "first_content_p95_ms": percentile([row["first_ms"] for row in rows], 0.95),
        "complete_p95_ms": percentile([row["complete_ms"] for row in rows], 0.95),
        "last_content_p95_ms": percentile(
            [row["last_content_ms"] for row in rows], 0.95
        ),
        "prep_p95_ms": percentile([row["prepare_ms"] for row in rows], 0.95),
        "prompt_tokens_min": min(inputs),
        "prompt_tokens_max": max(inputs),
        "prompt_tokens_p50": percentile(inputs, 0.50),
        "completion_tokens_p50": percentile(
            [m["usage"]["completion_tokens"] for m in measurements], 0.50
        ),
    }


def verify(root):
    arms = {mode: load(root / mode / "results.json") for mode in ("page", "fixed")}
    before, after = (
        load(root / f"lease-{stage}.json") for stage in ("before", "after")
    )
    assert before["pid"] == after["pid"] and before["gpu"] == after["gpu"]
    assert before["physical_index"] == after["physical_index"] == 6
    assert all(
        x["status"] == "running" and x["phase"] == "idle" and not x["other_processes"]
        for x in (before, after)
    )
    cases = list(corpus_cases())
    per_case = {}
    matched = 0
    for name, question, docs, expected in cases:
        per_case[name] = {}
        for technique in ("whole", "ranked"):
            selected = retrieve(question, docs) if technique == "ranked" else docs
            wanted_ids = {x.source for x in selected}
            body_hash = fingerprint(
                search_request("m", question, selected, token_witness=True)
            )
            page, fixed = (arms[mode]["search"][name][technique] for mode in arms)
            assert page["body_sha256"] == fixed["body_sha256"] == body_hash, name
            left, right = (r["measurement"] for r in (page, fixed))
            assert left["prompt_ids_sha256"] == right["prompt_ids_sha256"], name
            assert left["usage"]["prompt_tokens"] == right["usage"]["prompt_tokens"], (
                name
            )
            matched += 1
            per_case[name][technique] = {}
            for mode, row in zip(arms, (page, fixed), strict=True):
                m = row["measurement"]
                issue = grade(row["answer"], expected)
                assert row["corpus_sha256"] == fingerprint(
                    [(p.source, p.text) for p in docs]
                ), name
                assert row["problems"] == json.loads(json.dumps(issue)), name
                assert row["expected"] == expected, name
                assert row["correct"] == (
                    not any(issue.values()) and m["finish"] == "stop"
                ), name
                assert set(row["target_coverage"]) == (wanted_ids & expected.keys()), (
                    name
                )
                assert row["evidence_count"] == len(selected), name
                if technique == "ranked":
                    assert row["evidence_sources"] == [p.source for p in selected], name
                assert (
                    0 < row["first_ms"] <= row["last_content_ms"] <= row["complete_ms"]
                ), name
                assert (
                    not m["usage"]
                    .get("prompt_tokens_details", {})
                    .get("cached_tokens", 0)
                ), name
                per_case[name][technique][mode] = {
                    "correct": row["correct"],
                    "problems": row["problems"],
                }
    summaries = {
        mode: {
            technique: summarize([arm["search"][name][technique] for name, *_ in cases])
            for technique in ("whole", "ranked")
        }
        for mode, arm in arms.items()
    }
    legacy = {}
    for name, page in arms["page"]["diagnostics"].items():
        fixed = arms["fixed"]["diagnostics"][name]
        assert page["body_sha256"] == fixed["body_sha256"], name
        assert (
            page["measurement"]["prompt_ids_sha256"]
            == fixed["measurement"]["prompt_ids_sha256"]
        ), name
        legacy[name] = {
            mode: {
                "answer": row["answer"],
                "correct": row["correct"],
                "prompt_tokens": row["measurement"]["usage"]["prompt_tokens"],
                "finish": row["measurement"]["finish"],
                "complete_ms": row["complete_ms"],
            }
            for mode, row in (("page", page), ("fixed", fixed))
        }
    return {
        "device": root.joinpath("device.csv").read_text().splitlines()[1],
        "identity_sha256": root.joinpath("identity.sha256").read_text().splitlines(),
        "matched_prompt_pairs": matched,
        "summaries": summaries,
        "legacy": legacy,
        "per_case": per_case,
        "gpu_samples": {
            mode: check_shutdown(root, mode, before["pid"]) for mode in arms
        },
        "lease_restored": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args.directory), indent=2) + "\n")
