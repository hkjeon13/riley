"""Rebuild selected catalog workflows and compare actual final model input token by token."""

import argparse
import copy
import json
import subprocess
from functools import partial
from hashlib import sha256
from pathlib import Path
from statistics import median

from agent_tool_plan import resolve_named_step
from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    replay,
    turn_request,
)
from coder_catalog_trial import tool_output
from verify_model_decode import read


def check_fallback():
    tools = fixture(0, 2)[0]
    choice = {"type": "function", "function": {"name": "read_reports"}}
    assert resolve_named_step(tools, choice) == (
        "read_reports",
        {"first": "report-A", "second": "report-B"},
    )
    for ambiguous in (
        "auto",
        "required",
        "none",
        None,
        {"type": "function", "function": {"name": "not-a-tool"}},
    ):
        assert resolve_named_step(tools, ambiguous) is None
    assert resolve_named_step(tools + [tools[1]], choice) is None
    for change in (
        "optional",
        "multiple",
        "open",
        "constraint",
        "wrong_type",
        "duplicate_required",
        "bad_required",
        "bad_fields",
    ):
        t = copy.deepcopy(tools)
        schema = t[1]["function"]["parameters"]
        if change == "optional":
            schema["required"].pop()
        if change == "multiple":
            schema["properties"]["first"]["enum"].append("other")
        if change == "open":
            schema["additionalProperties"] = True
        if change == "constraint":
            schema["properties"]["first"]["pattern"] = "^forbidden$"
        if change == "wrong_type":
            schema["properties"]["first"]["enum"] = [None]
        if change == "duplicate_required":
            schema["required"] = ["first", "first"]
        if change == "bad_required":
            schema["required"] = [{}]
        if change == "bad_fields":
            schema["properties"] = []
        assert resolve_named_step(t, choice) is None


def capture_final(work, reader, *, on_tool_start=None):
    captured = []
    receipts = []

    def capture(address, request, row, first_visible):
        assert request["tool_choice"] == "none"
        captured.append(request)
        raise OSError("offline capture at final submission")

    plan = replay(
        "offline",
        "m",
        work["seed"],
        2,
        0,
        work["workflow_id"],
        history=work["history_entries"],
        send=capture,
        on_tool_start=on_tool_start,
        execute_tool=partial(tool_output, "selected", reader, work["seed"], receipts),
        dispatch_fixed_tools=True,
    )
    assert plan["error"] == "offline capture at final submission" and len(captured) == 1
    assert [t.get("dispatch") for t in plan["turns"]] == [
        "application",
        "application",
        None,
    ]
    return captured[0], plan["turns"][:2], receipts


def check(root, tokenizer, executable):
    check_fallback()
    digests = {
        name: sha256((tokenizer / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    assert list(digests.values()) == [
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4",
    ]
    assert not (tokenizer / "chat_template.jinja").exists()
    results = {}
    for gpu in (6, 7):
        rows = []
        requests = []
        for w in read(root / f"gpu{gpu}" / "selected" / "waves.json"):
            for j in w["jobs"]:
                work = j["work"]
                seed = work["seed"]
                reader = j["role"] == "reader"
                tools, prompt, docs = fixture(seed, 2)
                observed = history_messages(seed, work["history_entries"]) + [
                    {"role": "user", "content": prompt}
                ]
                planned, app_steps, receipts = capture_final(work, reader)
                calls = []
                for i, name in enumerate(("list_reports", "read_reports")):
                    turn = work["turns"][i]
                    assert (
                        fingerprint(turn_request("m", observed, tools, name))
                        == turn["input_sha256"]
                    )
                    assert (
                        app_steps[i]["call"]["function"]["name"]
                        == turn["call"]["function"]["name"]
                        == name
                    )
                    assert json.loads(
                        app_steps[i]["call"]["function"]["arguments"]
                    ) == json.loads(turn["call"]["function"]["arguments"])
                    append_tool_result(observed, [turn["call"]], docs, i)
                    calls.append(
                        {
                            "name": name,
                            "model_tokens": turn["usage"]["completion_tokens"],
                            "model_prompt_tokens": turn["usage"]["prompt_tokens"],
                            "model_elapsed_ms": (turn["done_ns"] - turn["sent_ns"])
                            / 1e6,
                        }
                    )
                assert receipts == j["tool_result_receipts"]
                final = work["turns"][2]
                ref = turn_request("m", observed, tools, None)
                assert fingerprint(ref) == final["input_sha256"]
                assert [
                    m["content"] for m in ref["messages"] if m["role"] == "tool"
                ] == [m["content"] for m in planned["messages"] if m["role"] == "tool"]
                requests.extend([ref, planned])
                rows.append(
                    {
                        "seed": seed,
                        "role": j["role"],
                        "wave": w["size"],
                        "turns": calls,
                        "served_final_prompt_tokens": final["usage"]["prompt_tokens"],
                    }
                )
        run = subprocess.run(
            [executable, tokenizer],
            input="".join(json.dumps(q) + "\n" for q in requests),
            text=True,
            capture_output=True,
            check=True,
        )
        tokens = iter(json.loads(line)["ids"] for line in run.stdout.splitlines())
        for r in rows:
            actual, next_input = next(tokens), next(tokens)
            assert len(actual) == r["served_final_prompt_tokens"]
            r.update(
                same_final_tokens=actual == next_input,
                actual_ids_sha256=fingerprint(actual),
                planned_ids_sha256=fingerprint(next_input),
            )
        assert next(tokens, None) is None
        targets = [r for r in rows if r["wave"] == 16 and r["role"] == "target"]
        results[str(gpu)] = {
            "jobs": len(rows),
            "same_tokens": sum(r["same_final_tokens"] for r in rows),
            "target_prior_model_wall_median_ms": median(
                sum(t["model_elapsed_ms"] for t in r["turns"]) for r in targets
            ),
            "rows": rows,
        }
    return {"tokenizer": digests, "gpus": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("raw", "tokenizer", "executable", "output"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    a.output.write_text(
        json.dumps(check(a.raw, a.tokenizer, a.executable), indent=2) + "\n"
    )
