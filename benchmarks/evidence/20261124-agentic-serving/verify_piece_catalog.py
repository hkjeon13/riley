"""Replay both tokenizers on the exact archived catalog requests and verify all IDs."""

import argparse
import json
from collections import Counter
from functools import partial
from hashlib import sha256
from pathlib import Path

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    turn_request,
)
from coder_catalog_trial import tool_output
from coder_piece_cost import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "baseline", "candidate", "tokenizer", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    prompts = {}
    counts = {}
    coverage = Counter()
    for gpu in (6, 7):
        for mode in ("full", "selected"):
            waves = json.loads((a.root / f"gpu{gpu}" / mode / "waves.json").read_text())
            for wave in waves:
                for job in wave["jobs"]:
                    reader = job["role"] == "reader"
                    work = job["work"]
                    seed = work["seed"]
                    tools, prompt, docs = fixture(seed, work["document_pairs"])
                    messages = history_messages(seed, work["history_entries"]) + [
                        {"role": "user", "content": prompt}
                    ]
                    for turn, (record, name) in enumerate(
                        zip(
                            work["turns"],
                            ["list_reports", "read_reports", None],
                            strict=True,
                        )
                    ):
                        req = turn_request("m", list(messages), tools, name)
                        key = fingerprint(req)
                        assert key == record["input_sha256"]
                        n = record["usage"]["prompt_tokens"]
                        if key in counts:
                            assert counts[key] == n
                        counts[key] = n
                        prompts[key] = req
                        coverage[(gpu, mode, turn)] += 1
                        if name is not None:
                            append_tool_result(
                                messages,
                                [record["call"]],
                                docs,
                                turn,
                                partial(tool_output, mode, reader, seed, []),
                            )
    keys = list(prompts)
    result_digest = sha256()
    for start in range(0, len(keys), 32):
        chunk = keys[start : start + 32]
        base = run(a.baseline, a.tokenizer, prompts, chunk)
        candidate = run(a.candidate, a.tokenizer, prompts, chunk)
        for k, old, new in zip(chunk, base, candidate, strict=True):
            assert old["ids"] == new["ids"] and len(old["ids"]) == counts[k]
            result_digest.update(bytes.fromhex(k))
            result_digest.update(bytes.fromhex(fingerprint(old["ids"])))
    assert len(coverage) == 12 and set(coverage.values()) == {44}
    a.out.write_text(
        json.dumps(
            {
                "observed_submissions": sum(coverage.values()),
                "distinct_http_requests": len(keys),
                "distinct_prompt_tokens": sum(counts.values()),
                "request_and_token_sha256": result_digest.hexdigest(),
                "per_group": [
                    {"gpu": g, "mode": m, "turn": t, "submissions": n}
                    for (g, m, t), n in sorted(coverage.items())
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
