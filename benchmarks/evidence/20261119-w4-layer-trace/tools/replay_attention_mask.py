"""Replay layer-0 attention from a logged Qwen KV page table and GPU snapshots."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def bf16(payload):
    words = np.frombuffer(payload, dtype="<u2")
    return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def capture(directory, expected_last):
    data = (directory / "rows.bf16").read_bytes()
    head, *index = (directory / "index.tsv").read_text().splitlines()
    marker, position, _, row, active, dense = head.split()
    if marker != "#" or int(position) != expected_last or not 0 < int(row) == int(active) - 1 < int(dense):
        raise ValueError("invalid final prefill row")
    rows = {}
    end = 0
    for line in index:
        stage, layer_or_page, offset, size = line.split("\t")
        key = (stage, layer_or_page)
        length = int(size)
        if key in rows or int(offset) != end or length <= 0 or length % 2:
            raise ValueError("invalid data extent")
        rows[key] = data[end:end + length]
        end += length
    if end != len(data):
        raise ValueError("truncated activation data")
    head, *pages = (directory / "kv-pages.tsv").read_text().splitlines()
    marker, first, last, page_size, head_width, kv_head = head.split()
    first, last = int(first), int(last)
    if (marker, last, page_size, head_width, kv_head) != ("#", expected_last, "16", "128", "0"):
        raise ValueError("unexpected attention geometry")
    if first != last - int(row) or len(pages) != (last // 16) + 1:
        raise ValueError("causal extent or page count differs from the final prefill metadata")
    keys, values, physical = [], [], []
    for ordinal, line in enumerate(pages):
        logical, block, valid, used = map(int, line.split("\t"))
        expected = min(16, last + 1 - ordinal * 16)
        if logical != ordinal or block < 0 or not used == expected <= valid <= 16:
            raise ValueError("invalid KV page extent")
        physical.append(block)
        for name, vectors in (("kv_key", keys), ("kv_value", values)):
            payload = rows[(name, str(ordinal))]
            if len(payload) != used * 128 * 2:
                raise ValueError("invalid KV page snapshot")
            vectors.append(bf16(payload).reshape(-1, 128))
    if len(set(physical)) != len(physical):
        raise ValueError("physical pages alias within the single request")
    return rows, first, last, physical, np.concatenate(keys), np.concatenate(values)


def attention(query, key, value):
    scores = query @ key.T / np.sqrt(128)
    weighted = np.exp(scores - scores.max(axis=1, keepdims=True))
    return (weighted / weighted.sum(axis=1, keepdims=True)) @ value


def compare(actual, expected):
    delta = actual - expected
    rms = np.sqrt(np.mean(delta * delta, axis=1))
    return {"rms": float(np.sqrt(np.mean(delta * delta))),
            "max_abs": float(np.abs(delta).max()),
            "per_query_head_rms": rms.tolist()}


def replay(directory, fixture, case):
    original = json.loads(fixture.read_bytes())["baseline"][case]
    saved = json.loads((directory / "response.json").read_bytes())
    if (not saved["all_original_ids_text_and_finish"]
            or saved["response"]["prompt_tokens"] != original["prompt_tokens"]):
        raise ValueError("instrumented output differs from the pinned source")
    rows, first, last, pages, keys, values = capture(directory, len(original["prompt_tokens"]) - 1)
    probes = {}
    for position, suffix in ((first, "_first"), (last, "")):
        query = bf16(rows[(f"query_rope{suffix}", "0")]).reshape(28, 128)[:7]
        actual = bf16(rows[(f"attention_context{suffix}", "0")]).reshape(28, 128)[:7]
        stop = position + 1
        choices = {"causal": (0, stop), "omit_first_page": (16, stop),
                   "omit_last_visible_token": (0, stop - 1)}
        if position != last:
            choices["include_future_tokens"] = (0, last + 1)
        probes[str(position)] = {name: compare(actual, attention(query, keys[a:b], values[a:b]))
                                 for name, (a, b) in choices.items()}
    return {"case": case, "layer": 0, "kv_head": 0, "query_heads": 7,
            "first_position": first, "final_position": last, "physical_pages": pages,
            "prompt_sha256": original["prompt_sha256"],
            "page_data_sha256": sha256((directory / "rows.bf16").read_bytes()).hexdigest(),
            "reference": "Float64 softmax over native BF16 Q/K/V using logical token visibility",
            "probes": probes}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("case", type=int, choices=(3, 9))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = replay(args.directory, args.fixture, args.case)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
