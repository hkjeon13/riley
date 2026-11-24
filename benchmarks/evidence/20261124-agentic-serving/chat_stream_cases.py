"""Reconstruct the four A100 Coder texts and validate the hashes of each natural SSE replay."""

import argparse
import json
from pathlib import Path

from agent_workflow import fingerprint
from tokenizers import Tokenizer


def cases(path, tokenizer_dir):
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    observed = json.loads(path.read_text())
    inputs, metadata = [], []
    for row in observed:
        if row["phase"] != "identity":
            continue
        ref = row["result"]
        ids = ref["generated_ids"]
        pieces = [tokenizer.decode([token], skip_special_tokens=True) for token in ids]
        assert len(pieces) == 64 and fingerprint("".join(pieces)) == ref["text_sha256"]
        times = [
            run["tpot_ms"]
            for other in observed
            if other["phase"] == "normal" and other["history"] == row["history"]
            for run in other["results"]
            if run["name"] == ref["name"]
            and run["text_sha256"] == ref["text_sha256"]
            and len(run["content_ns"]) == len(pieces)
        ]
        assert len(times) == 3
        inputs.append({"style": None, "pieces": pieces})
        metadata.append(
            {
                "request": ref["name"],
                "history": row["history"],
                "sha256": ref["text_sha256"],
                "a100_normal_tpot_ms": times,
            }
        )
    assert len(inputs) == 4
    return inputs, metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("record", "tokenizer", "cases", "metadata"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    inputs, metadata = cases(args.record, args.tokenizer)
    inputs.extend(
        [
            {
                "style": "hermes",
                "pieces": [
                    "측정값 ",
                    "42 ms.\n",
                    "<tool_",
                    'call>{"name":"read","arguments":{"id":"A"}}',
                    "</tool_call>",
                ],
            },
            {
                "style": "llama",
                "pieces": [
                    " ",
                    "<|py",
                    'thon_tag|>{"name":"read",',
                    '"parameters":{"id":"A"}}',
                ],
            },
            {
                "style": "hermes",
                "pieces": ["😀", " before ", "<tool_", "rest", "<tool_call>broken"],
            },
        ]
    )
    args.cases.write_text(json.dumps(inputs, ensure_ascii=False) + "\n")
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
