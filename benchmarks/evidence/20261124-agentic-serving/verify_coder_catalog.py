"""Reconstruct all catalog tool results and pair the scheduled workload across arms."""

import argparse
import json
import re
from functools import partial
from pathlib import Path

from agent_tool_projection import project_exact_records
from agent_workflow import fingerprint, fixture, fixture_tool_result
from coder_catalog_trial import catalog
from verify_coder_pressure import check_manifest, check_server, natural, read
from verify_http_worker_queue import summary, verify_wave


def verify_job(mode, job, seed, reader, *, application_tools=False):
    expected = []

    def execute(turn, call, docs):
        if reader or turn == 0:
            return fixture_tool_result(turn, docs)
        arguments = json.loads(call["function"]["arguments"])
        keys = [arguments["first"], arguments["second"]]
        assert keys == ["report-A", "report-B"]
        rows = catalog(seed, docs)
        assert len({r["id"] for r in rows}) == len(rows) == 64
        selected = [r for r in rows if r["id"] in keys]
        assert selected == docs
        result = rows if mode == "full" else selected
        expected.append(
            {
                "source_sha256": fingerprint(rows),
                "result_sha256": fingerprint(result),
                "source_count": len(rows),
                "result_count": len(result),
                "requested_ids": keys,
            }
        )
        return result

    facts = natural(
        job["work"],
        seed,
        2,
        0 if reader else 200,
        delay=0 if reader else 0.5,
        execute_tool=execute,
        application_tools=application_tools,
    )
    assert job["tool_result_receipts"] == expected
    return facts


def answer_checks(work):
    final = work["turns"][-1]
    docs = fixture(work["seed"], 2)[2]
    a, b = [r["p95_ms"] for r in docs]
    text = final.get("answer", "").replace("*", "").lower()
    return {
        "all_three_values": all(
            re.search(rf"\b{label}\s*:\s*{v}\s*ms\b", text)
            for label, v in zip(("before", "after", "reduction"), (a, b, a - b))
        ),
        "both_sources": all(doc["id"].lower() in text for doc in docs),
        "at_least_64_tokens": final.get("usage", {}).get("completion_tokens", 0) >= 64,
    }


def verify(root):
    check_manifest(root)
    cfg = read(root / "coder-catalog-registration.json")
    sha = (root / "trace-server.sha256").read_text().split()[0]
    assert sha == "5262f4450c62de25c68d979978627ca6715b8cd4d642cb0abcbe6dacbc6a291c"
    result = {}
    cli = None
    no_op = 0
    for size, base in cfg["waves"]:
        for i in range(size):
            if i % 4:
                docs = fixture(base + i, 96)[2]
                assert (
                    project_exact_records(
                        docs, key="id", requested_ids=["report-A", "report-B"]
                    )
                    == docs
                )
                no_op += 1
    for gpu, order in cfg["orders"].items():
        arms = result[gpu] = {}
        for mode in order:
            d = root / f"gpu{gpu}" / mode
            check_server(d, sha, gpu)
            assert read(d / "warm.json")["ok"]
            args = (d / "server-arguments.txt").read_text()
            if cli is None:
                cli = args
            assert cli == args
            waves = []
            for w, (size, base) in zip(
                read(d / "waves.json"), cfg["waves"], strict=True
            ):
                v = verify_wave(
                    w,
                    (d / "server.log").read_bytes(),
                    size,
                    base,
                    check_job=partial(verify_job, mode),
                )
                for row, job in zip(v["rows"], w["jobs"], strict=True):
                    row.update(answer_checks(job["work"]))
                waves.append(v)
            arms[mode] = {
                "by_wave": [summary([w]) for w in waves],
                "high": summary(waves[2:]),
                "waves": waves,
            }
            long_rows = [
                r for w in waves[2:] for r in w["rows"] if r["role"] == "target"
            ]
            complete = [r for r in long_rows if r.get("completed")]
            arms[mode]["high"]["target_submitted_after_first_deadline"] = sum(
                r["before_final_submission_ms"] >= 5000 for r in complete
            )
            arms[mode]["high"]["target_answer_checks"] = {
                k: sum(r[k] for r in long_rows)
                for k in ("all_three_values", "both_sources", "at_least_64_tokens")
            }
        a, b = [arms[m]["high"] for m in ("full", "selected")]
        arms["decision"] = {
            "more_slo": b["slo"] > a["slo"],
            "more_goodput": b["warm_slo_per_gpu_hour"] > a["warm_slo_per_gpu_hour"],
            "better_last": b["last_p95_ms"] < a["last_p95_ms"],
            "less_prefill": b["target_prefill_by_turn"][-1]["executed_tokens"]
            < a["target_prefill_by_turn"][-1]["executed_tokens"],
            "preserve_valid": set(b["invalid_seeds"]) <= set(a["invalid_seeds"]),
            **{
                k: b[k] <= 1.05 * a[k]
                for k in ("reader_first_p95_ms", "reader_gap_p95_ms")
            },
        }
    return {"two_report_noop_samples": no_op, "gpus": result}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.write_text(json.dumps(verify(a.root), indent=2, allow_nan=False) + "\n")
