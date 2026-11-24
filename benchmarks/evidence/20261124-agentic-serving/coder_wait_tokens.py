"""Compute shared complete KV blocks by independently rendering the three actual request bodies."""

import argparse
import copy
import json
from pathlib import Path

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    turn_request,
)
from transformers import AutoTokenizer


def tokenize(tok, messages):
    msgs = copy.deepcopy(messages)
    for message in msgs:
        for call in message.get("tool_calls", []):
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(rendered, add_special_tokens=False)["input_ids"]


def compare(tok, work, identity):
    seed = work["seed"]
    tools, prompt, docs = fixture(seed, work["document_pairs"])
    history = history_messages(seed, work["history_entries"]) + [
        {"role": "user", "content": prompt}
    ]
    append_tool_result(history, [work["turns"][0]["call"]], docs, 0)
    call = work["turns"][1]["call"]
    full = list(history)
    append_tool_result(full, [call], docs, 1)
    final = tokenize(tok, full)
    final_row = work["turns"][-1]
    assert (
        fingerprint(turn_request("m", full, tools, None))
        == final_row["input_sha256"]
        == identity["input"]
    )
    assert (
        fingerprint(final) == identity["tokens"]
        and len(final) == final_row["usage"]["prompt_tokens"]
    )
    out = {}
    for mode, hand in (
        ("history", history),
        ("boundary", [*full[:-1], {**full[-1], "content": ""}]),
    ):
        preview = tokenize(tok, hand)
        shared = next(
            (i for i, (a, b) in enumerate(zip(preview, final)) if a != b),
            min(len(preview), len(final)),
        )
        out[mode] = {
            "tokens": len(preview),
            "shared": shared,
            "shared_complete_tokens": (min(len(final) - 1, shared) // 16) * 16,
        }
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("--previous", action="store_true")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.previous:
        ids = json.loads((args.root / "identity.json").read_text())
        rows = json.loads((args.root / "natural.json").read_text())
        ids = {
            s: {"input": v["original_input_sha256"], "tokens": v["prompt_ids_sha256"]}
            for s, v in ids.items()
        }
    else:
        ids = json.loads((args.root / "identity.json").read_text())
        rows = [r["target"] for r in json.loads((args.root / "cases.json").read_text())]
    print(
        json.dumps(
            {w["seed"]: compare(tok, w, ids[str(w["seed"])]) for w in rows}, indent=2
        )
    )
