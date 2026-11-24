"""Verify real cached CPU forward length, recorded tool times, complete answer IDs and text."""

import argparse
import json
import subprocess
from pathlib import Path

from agent_workflow import fingerprint, fixture, history_messages, turn_request
from tool_wait_trial import prepare_prefix
from verify_named_cpu_live import verify as verify_generation


def load(path):
    return json.loads(path.read_text())


def verify(root, base, profiler, tokenizer):
    checked = verify_generation(root, profiler, tokenizer)
    control = verify_generation(base, profiler, tokenizer)
    by_seed = {
        j["seed"]: (j, i)
        for j, i in zip(control["jobs"], load(base / "inference.json"), strict=True)
    }
    cfg = load(root / "named-cpu-registration.json")["first_tool_kv"]
    summary = []
    for job, client, obs, prep in zip(
        checked["jobs"],
        load(root / "client.json"),
        load(root / "inference.json"),
        load(root / "preparation.json"),
        strict=True,
    ):
        work = client["work"]
        other, baseline = by_seed[job["seed"]]
        assert checked["engine"] == control["engine"]
        assert obs["prompt_ids_sha256"] == baseline["prompt_ids_sha256"]
        assert (
            obs["output_ids"] == baseline["output_ids"]
            and job["answer"] == other["answer"]
        )
        assert (
            job["finish"] == other["finish"] == "stop"
            and job["usage"] == other["usage"]
        )
        assert len(work["turns"]) == 3 and "error" not in prep
        first, second, final = work["turns"]
        assert (
            first["tool_start_ns"]
            <= prep["decision_ns"]
            <= first["tool_ready_ns"]
            <= second["tool_start_ns"]
            <= second["tool_ready_ns"]
            <= final["sent_ns"]
        )
        assert (
            final["sent_ns"]
            <= prep["join_started_ns"]
            <= prep["joined_ns"]
            <= final["content_ns"][0]
        )
        assert (
            job["usage"]["prompt_tokens"] - prep["reused"]
            == prep["first_forward_tokens"]
        )
        assert prep["final_tokens"] == job["usage"]["prompt_tokens"]
        committed = 0
        before = 0
        for start, end, count in prep["commits"]:
            assert prep["decision_ns"] <= start < end <= prep["joined_ns"]
            assert count - committed == cfg["chunk"]
            if "wait_hint_ms" in cfg:
                assert (
                    start + cfg["chunk_budget_ms"] * 10**6
                    <= prep["decision_ns"] + cfg["wait_hint_ms"] * 10**6
                )
            committed = count
            if end <= final["sent_ns"]:
                before = count
        assert committed == prep["computed"] <= cfg["maximum_tokens"]
        if prep["decision"] == "prepare":
            tools, prompt, _ = fixture(work["seed"], work["document_pairs"])
            msgs = history_messages(work["seed"], work["history_entries"]) + [
                {"role": "user", "content": prompt}
            ]
            partial = prepare_prefix(
                turn_request("m", msgs, tools, "list_reports"),
                first["call"],
                include_completed_call=True,
            )
            assert partial == prep["request"] and prep["input_sha256"] == fingerprint(
                partial
            )
            assert [
                m["content"] for m in partial["messages"] if m["role"] == "tool"
            ] == [""]
            from agent_workflow import append_tool_result

            docs = fixture(work["seed"], work["document_pairs"])[2]
            for i, t in enumerate((first, second)):
                append_tool_result(msgs, [t["call"]], docs, i)
            complete = turn_request("m", msgs, tools, None)
            assert fingerprint(complete) == final["input_sha256"]
            pro = subprocess.run(
                [profiler, tokenizer],
                input=json.dumps(partial) + "\n" + json.dumps(complete) + "\n",
                capture_output=True,
                text=True,
                check=True,
            )
            early, late = [json.loads(line)["ids"] for line in pro.stdout.splitlines()]
            shared = next(
                (i for i, (a, b) in enumerate(zip(early, late)) if a != b),
                min(len(early), len(late)),
            )
            assert (
                fingerprint(early) == prep["preview_ids_sha256"]
                and fingerprint(late) == obs["prompt_ids_sha256"]
            )
            assert shared == prep["common_tokens"] and committed <= len(early)
            assert (
                prep["reused"]
                == min(shared, len(late) - 1, committed) // cfg["block"] * cfg["block"]
            )
        else:
            assert (
                prep["decision"] == "short_context" and committed == prep["reused"] == 0
            )
        summary.append(
            {
                "seed": job["seed"],
                "prepared": committed,
                "committed_by_final_submission": before,
                "reused": prep["reused"],
                "first_forward_tokens": prep["first_forward_tokens"],
                "input_tokens": prep["final_tokens"],
                "output_tokens": len(obs["output_ids"]),
                "output_ids_equal": True,
                "both_sources": job["both_sources"],
                "all_three_values": job["all_three_values"],
                "finish": job["finish"],
                "join_ms": (prep["joined_ns"] - prep["join_started_ns"]) / 1e6,
                "remaining_after_submission_ms": max(
                    0, prep["commits"][-1][1] - final["sent_ns"]
                )
                / 1e6
                if committed
                else 0,
                "stop_reason": prep.get("stop_reason"),
            }
        )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "control", "profiler", "tokenizer", "out"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    a.out.write_text(
        json.dumps(verify(a.root, a.control, a.profiler, a.tokenizer), indent=2) + "\n"
    )
