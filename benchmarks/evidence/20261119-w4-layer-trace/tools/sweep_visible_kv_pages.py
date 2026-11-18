"""Measure how the captured attention outputs distinguish individual visible pages."""

from argparse import ArgumentParser
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from replay_attention_mask import attention, bf16, capture


def rms(delta):
    return float(np.sqrt(np.mean(np.square(delta))))


def sweep_row(position, query, actual, keys, values):
    stop = position + 1
    keys, values = keys[:stop], values[:stop]
    scores = query @ keys.T / np.sqrt(128)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    complete = attention(query, keys, values)
    complete_error = rms(actual - complete)
    omissions = []
    for start in range(0, stop, 16):
        end = min(start + 16, stop)
        kept_weights = np.concatenate((weights[:, :start], weights[:, end:]), axis=1)
        kept_values = np.concatenate((values[:start], values[end:]))
        total_weight = kept_weights.sum(axis=1, keepdims=True)
        if np.any(total_weight == 0):
            kept_keys = np.concatenate((keys[:start], keys[end:]))
            omitted = attention(query, kept_keys, kept_values)
        else:
            omitted = (kept_weights / total_weight) @ kept_values
        omissions.append({"logical_page": start // 16, "removed_visible_tokens": end - start,
                          "observed_rms": rms(actual - omitted), "change_rms": rms(complete - omitted)})
    return {"position": position, "complete_rms": complete_error,
            "complete_closer": sum(x["observed_rms"] > complete_error for x in omissions),
            "omissions": omissions}


def sweep(root):
    checksums = dict(line.split(maxsplit=1)[::-1] for line in (root / "SHA256SUMS").read_text().splitlines())
    inputs = ["fixtures/legacy.json.gz"] + [
        f"native{case}/{name}" for case in (3, 9)
        for name in ("response.json", "index.tsv", "kv-pages.tsv", "rows.bf16.gz")
    ]
    for name in inputs:
        if sha256((root / name).read_bytes()).hexdigest() != checksums[name]:
            raise ValueError(f"checksum disagrees for {name}")
    fixture = json.loads(gzip.decompress((root / inputs[0]).read_bytes()))["baseline"]
    cases = []
    for case in (3, 9):
        directory = root / f"native{case}"
        if (directory / "rows.bf16").exists():
            raise ValueError("unverified uncompressed capture would take precedence")
        original = fixture[case]
        saved = json.loads((directory / "response.json").read_bytes())
        if (not saved["all_original_ids_text_and_finish"]
                or saved["response"]["prompt_tokens"] != original["prompt_tokens"]):
            raise ValueError(f"captured output differs from the pinned source for case {case}")
        rows, first, last, _, keys, values, _ = capture(directory, len(original["prompt_tokens"]) - 1)
        probes = []
        for position, suffix in ((first, "_first"), (last, "")):
            query = bf16(rows[(f"query_rope{suffix}", "0")]).reshape(28, 128)[:7]
            actual = bf16(rows[(f"attention_context{suffix}", "0")]).reshape(28, 128)[:7]
            probes.append(sweep_row(position, query, actual, keys, values))
        cases.append({"case": case, "positions": probes})
    return {"reference": "FP64 attention over captured native BF16 Q/K/V", "layer": 0, "kv_head": 0,
            "query_heads": 7, "input_sha256": {name: checksums[name] for name in inputs}, "cases": cases}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = sweep(args.root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
