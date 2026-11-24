"""Count exact precomputable, prefix-cache-addressable blocks before each tool returns."""

import argparse
import itertools
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
from tool_wait_trial import prepare_prefix
from verify_coder_named_steps import capture_final


def common(a, b):
    return next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )


def capacity(paths, limit):
    # Represent the actual cache key (parent, 16 token IDs) with an injective local index.
    index = {}
    masks = []
    for path in paths:
        parent = -1
        mask = 0
        for tokens in path:
            parent = index.setdefault((parent, tokens), len(index))
            mask |= 1 << parent
        masks.append(mask)
    # Small registered waves have at most twelve long jobs. Exhaustively count
    # how many full prepared prefixes could coexist, granting perfect sharing.
    best = len(masks)
    while best > 0 and not any(
        union(parts).bit_count() <= limit
        for parts in itertools.combinations(masks, best)
    ):
        best -= 1
    return {
        "private_blocks": sum(map(len, paths)),
        "distinct_blocks": len(index),
        "whole_prefixes_fit": best,
        "targets": len(paths),
        "limit": limit,
    }


def union(parts):
    result = 0
    for part in parts:
        result |= part
    return result


def audit(root, profiler, tokenizer, conf):
    assert conf["torch_dtype"] == "bfloat16"
    bytes_per_block = (
        2
        * conf["num_hidden_layers"]
        * conf["num_key_value_heads"]
        * (conf["hidden_size"] // conf["num_attention_heads"])
        * 2
        * 16
    )
    results = {}
    for gpu in ("gpu6", "gpu7"):
        root_arm = root / gpu / "selected"
        args = (root_arm / "server-arguments.txt").read_text().splitlines()
        limit = int(args[args.index("--prefix-cache-blocks") + 1])
        pool = int(args[args.index("--kv-blocks") + 1])
        waves = []
        for wave in json.loads((root_arm / "waves.json").read_text()):
            queries = []
            for job in wave["jobs"]:
                work = job["work"]
                previews = []

                def snapshot(index, request, row, previews=previews):
                    assert index == len(previews) and row["dispatch"] == "application"
                    previews.append(
                        prepare_prefix(
                            request, row["call"], include_completed_call=True
                        )
                    )

                final, calls, receipts = capture_final(
                    work, job["role"] == "reader", on_tool_start=snapshot
                )
                assert receipts == job["tool_result_receipts"]
                assert len(previews) == 2
                tools, prompt, docs = fixture(work["seed"], work["document_pairs"])
                messages = history_messages(work["seed"], work["history_entries"]) + [
                    {"role": "user", "content": prompt}
                ]
                for i, (call, row) in enumerate(
                    zip(calls, work["turns"][:2], strict=True)
                ):
                    assert call["call"]["function"] == row["call"]["function"]
                    assert [
                        m["content"]
                        for m in previews[i]["messages"]
                        if m["role"] == "tool"
                    ] == ([""] if i == 0 else [json.dumps([d["id"] for d in docs]), ""])
                    append_tool_result(messages, [row["call"]], docs, i)
                archived = turn_request("m", messages, tools, None)
                assert fingerprint(archived) == work["turns"][2]["input_sha256"]
                queries.extend([*previews, final, archived])
            command = subprocess.run(
                [profiler, tokenizer],
                input="".join(json.dumps(q) + "\n" for q in queries),
                text=True,
                capture_output=True,
                check=True,
            )
            it = iter(json.loads(line)["ids"] for line in command.stdout.splitlines())
            rows = []
            paths = [[], []]
            for job, (first, second, final, archived) in zip(
                wave["jobs"], zip(it, it, it, it, strict=True), strict=True
            ):
                w = job["work"]
                assert (
                    final == archived
                    and len(archived) == w["turns"][2]["usage"]["prompt_tokens"]
                )
                counts = []
                for k, prefix in enumerate((first, second)):
                    safe = min(common(prefix, final), len(final) - 1) // 16
                    counts.append(safe * 16)
                    if job["role"] == "target":
                        paths[k].append(
                            [tuple(final[i : i + 16]) for i in range(0, 16 * safe, 16)]
                        )
                rows.append(
                    {
                        "seed": w["seed"],
                        "role": job["role"],
                        "final_tokens": len(final),
                        "prefix_tokens": counts,
                        "suffix_tokens": [len(final) - n for n in counts],
                    }
                )
            waves.append(
                {
                    "base": wave["base"],
                    "size": wave["size"],
                    "pool_blocks": pool,
                    "workers": int(args[args.index("--max-active-sequences") + 1]),
                    "stage_capacity": [capacity(p, limit) for p in paths],
                    "jobs": rows,
                }
            )
        results[gpu] = waves
    return {"kv_bytes_per_16_token_block": bytes_per_block, "devices": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "profiler", "tokenizer", "config", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    a.out.write_text(
        json.dumps(
            audit(a.root, a.profiler, a.tokenizer, json.loads(a.config.read_text())),
            indent=2,
        )
        + "\n"
    )
