"""Count final-turn KV blocks computable before the second tool returns."""

import argparse
import json
import subprocess
from pathlib import Path

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    turn_request,
)


def preview(work, model):
    tools, prompt, docs = fixture(work["seed"], work["document_pairs"])
    messages = history_messages(work["seed"], work.get("history_entries", 0)) + [
        {"role": "user", "content": prompt}
    ]
    append_tool_result(messages, [work["turns"][0]["call"]], docs, 0)
    # Only messages available before the second tool returns are included.
    before = turn_request(model, list(messages), tools, None)
    before["max_tokens"] = 1
    append_tool_result(messages, [work["turns"][1]["call"]], docs, 1)
    final = turn_request(model, messages, tools, None)
    if fingerprint(final) != work["turns"][2]["input_sha256"]:
        raise ValueError("final submission hash changed")
    return before, final


def audit(records, binary, tokenizer):
    requests = [r for w in records for r in preview(w, "m")]
    rows = subprocess.run(
        [binary, tokenizer],
        input="".join(json.dumps(r) + "\n" for r in requests),
        text=True,
        capture_output=True,
        check=True,
    )
    ids = [json.loads(line)["ids"] for line in rows.stdout.splitlines()]
    out = []
    for work, a, b in zip(records, ids[::2], ids[1::2], strict=True):
        row = work["turns"][2]
        if len(b) != row["usage"]["prompt_tokens"]:
            raise ValueError("exact tokenizer disagrees with live prompt length")
        shared = next(
            (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
        )
        eligible = min(shared, len(b) - 1) // 16 * 16
        actual = (row["usage"].get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        )
        out.append(
            {
                "workflow": work["workflow_id"],
                "history": work["history_entries"],
                "preview_tokens": len(a),
                "final_prompt_tokens": len(b),
                "actual_cached": actual,
                "preview_eligible": eligible,
                "additional_eligible": max(0, eligible - actual),
                "measured_tool_wait_ms": (
                    work["turns"][1]["tool_ready_ns"]
                    - work["turns"][1]["tool_start_ns"]
                )
                / 1e6,
            }
        )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflows", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("tokenizer", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                json.loads(args.workflows.read_text()), args.executable, args.tokenizer
            ),
            indent=2,
        )
    )
