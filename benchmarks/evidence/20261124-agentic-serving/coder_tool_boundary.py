"""Locate complete KV blocks before the second tool result in the actual rendered final prompt."""

import json
from pathlib import Path

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    turn_request,
)


def locate(record, identity, tokenizer):
    seed = record["seed"]
    tools, prompt, docs = fixture(seed, record["document_pairs"])
    msgs = history_messages(seed, record["history_entries"]) + [
        {"role": "user", "content": prompt}
    ]
    for i in (0, 1):
        append_tool_result(msgs, [record["turns"][i]["call"]], docs, i)
    assert (
        fingerprint(turn_request("m", msgs, tools, None))
        == identity["original_input_sha256"]
    )
    payload = msgs[-1]["content"]
    for msg in msgs:
        for call in msg.get("tool_calls", []):
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    text = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )
    assert text.count(payload) == 1
    start = text.index(payload)
    tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = tokens["input_ids"]
    assert (
        fingerprint(ids) == identity["prompt_ids_sha256"]
        and len(ids) == identity["prompt_tokens"]
    )
    pos = next(i for i, (a, b) in enumerate(tokens["offset_mapping"]) if b > start)
    cached = (record["turns"][-1]["usage"].get("prompt_tokens_details") or {}).get(
        "cached_tokens", 0
    )
    blocks = pos // 16 * 16
    return {
        "seed": seed,
        "prompt": len(ids),
        "cached": cached,
        "complete_tokens_before_payload": blocks,
        "additional_preparable": max(0, blocks - cached),
        "tokens_from_boundary_block_to_end": len(ids) - blocks,
        "prompt_ids_sha256": identity["prompt_ids_sha256"],
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    root = Path(__file__).resolve().parent
    model = root.parents[2] / "models" / "Qwen2.5-Coder-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    out = {}
    for gpu, mode in ((6, "page"), (7, "tile")):
        arm = root / f"gpu{gpu}" / mode
        records = json.loads((arm / "natural.json").read_text())
        identity = json.loads((arm / "identity.json").read_text())
        out[gpu] = [locate(r, identity[str(r["seed"])], tok) for r in records]
    assert out[6] == out[7]
    (root / "tool-boundary.json").write_text(json.dumps(out[6], indent=2) + "\n")
