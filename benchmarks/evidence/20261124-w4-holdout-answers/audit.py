#!/usr/bin/env python3
"""Compare the stored holdout completions before reviewing their answers."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parents[1]
sys.path.insert(0, str(BENCHMARKS / "live"))
from long_accuracy import (
    HOLDOUT_TOKENIZER_SHA256, common_prefix, decode_capture, read_workload, terminal_normalized, token_ids,
)

FIXTURE = HERE.parent / "20261119-w4-holdout"
PROJECTIONS = {
    "attention-plan-31": "w4-o31-serving-20261120.7W3nr5JH",
    "fp32-down-plan-21": "w4-fp32-down-serving-20261120.Saj3XXSY",
}


def checksums(path):
    return {name: digest for digest, name in (line.split() for line in path.read_text().splitlines())}


def read_verified(path, expected, *, uncompressed=False):
    raw = path.read_bytes()
    digest = sha256(gzip.decompress(raw) if uncompressed else raw).hexdigest()
    if digest != expected:
        raise ValueError(f"unexpected capture digest for {path}")
    return raw, {"path": str(path.relative_to(BENCHMARKS.parent)), "sha256": digest,
                 "digest_of": "uncompressed" if uncompressed else "file"}


def audit_text_tail(reference, attention):
    tail = json.loads((HERE / "tokenizer-tail.json").read_bytes())
    suffix_ids = reference["hf_tokens"][-len(tail["tokens"]):]
    if (tail["tokenizer_sha256"] != HOLDOUT_TOKENIZER_SHA256
            or attention["tokens"] != reference["hf_tokens"]):
        raise ValueError("tail does not come from matching generation IDs")
    visible = [*range(33, 127), *range(161, 173), *range(174, 256)]
    escaped = [byte for byte in range(256) if byte not in visible]
    alphabet = visible + list(range(256, 256 + len(escaped)))
    codepoint_to_byte = dict(zip(alphabet, visible + escaped, strict=True))
    encoded = "".join(tail["tokens"][str(i)] for i in suffix_ids)
    raw = bytes(codepoint_to_byte[ord(char)] for char in encoded)
    try:
        raw.decode()
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data" or error.end != len(raw):
            raise
        valid = raw[:error.start].decode()
        replaced = raw.decode(errors="replace")
        if (not attention["text"].endswith(valid) or not reference["hf_text"].endswith(replaced)
                or attention["text"] != reference["hf_text"][:-1]):
            raise ValueError("decoded suffix does not explain the captured text difference")
        return {"label": reference["label"], "suffix_token_ids": suffix_ids,
                "incomplete_terminal_bytes_hex": raw[error.start:].hex()}
    raise ValueError("expected an incomplete terminal UTF-8 sequence")


def audit():
    pinned = checksums(FIXTURE / "SHA256SUMS")
    projection_hashes = checksums(HERE.parent / "20261122-w4-projection-divergence-replay" / "captures.sha256")
    sources = {}
    fixture_path = FIXTURE / "holdout.json.gz"
    _, sources["workload"] = read_verified(fixture_path, pinned["./holdout.json.gz"])
    workload = read_workload(fixture_path)
    hf_raw, sources["hf-sdpa"] = read_verified(FIXTURE / "hf/reference.json", pinned["./hf/reference.json"])
    vllm_raw, sources["vllm"] = read_verified(FIXTURE / "vllm/results.json.gz", pinned["./vllm/results.json.gz"])
    hf = json.loads(hf_raw)
    vllm = json.loads(gzip.decompress(vllm_raw))
    if any(data["workload_sha256"] != workload.sha256 for data in (hf, vllm)):
        raise ValueError("reference workloads differ")
    riley_raw, sources["riley"] = read_verified(FIXTURE / "riley/capture.json.gz", pinned["./riley/capture.json.gz"])
    captures = {"riley": decode_capture(riley_raw, ".gz", workload)["baseline"]}
    for name, stage in PROJECTIONS.items():
        path = HERE / f"{name}.json.gz"
        raw, sources[name] = read_verified(
            path, projection_hashes[f"{stage}/candidate/rows/holdout.json"], uncompressed=True,
        )
        captures[name] = decode_capture(raw, path.suffix, workload)["baseline"]

    cases = {}
    hf_eos_ids = hf["eos_token_ids"]
    token_matches = dict.fromkeys(("riley", "vllm", *PROJECTIONS), 0)
    text_matches = token_matches.copy()
    text_tail = None
    for index, (case, ref, vllm_row) in enumerate(zip(workload.cases, hf["cases"], vllm["reference"], strict=True)):
        if (ref["case"] != index or ref["label"] != case.label
                or ref["prompt_tokens"] != len(case.expected_ids) or ref["hf_tokenizer_ids_match"] is not True
                or vllm_row["prompt_tokens"] != case.expected_ids):
            raise ValueError(f"reference input differs for {case.label}")
        reference = {"tokens": ref["hf_tokens"], "text": ref["hf_text"], "finish": ref["hf_finish"]}
        responses = {"riley": captures["riley"][index], "hf-sdpa": reference, "vllm": vllm_row,
                     **{name: captures[name][index] for name in PROJECTIONS}}
        if case.label == "korean-1024":
            text_tail = audit_text_tail(ref, responses["attention-plan-31"])
        normalized_ref = terminal_normalized(reference["tokens"], reference["finish"], hf_eos_ids)
        results = {}
        for name, row in responses.items():
            tokens, finish = token_ids(row["tokens"]), row["finish"]
            if (finish not in ("stop", "length") or len(tokens) > case.limit
                    or (finish == "length" and len(tokens) != case.limit)):
                raise ValueError(f"inconsistent completion length for {case.label} on {name}")
            normalized = terminal_normalized(tokens, finish, hf_eos_ids)
            match = normalized == normalized_ref and finish == reference["finish"]
            text_match = row["text"] == reference["text"]
            if name in token_matches:
                token_matches[name] += int(match)
                text_matches[name] += int(text_match)
            results[name] = {
                "completion_tokens": len(tokens),
                "finish": finish,
                "hf_exact_tokens_and_finish": match,
                "hf_common_prefix_tokens": common_prefix(normalized, normalized_ref),
                "hf_exact_text": text_match,
                "text_sha256": sha256(row["text"].encode()).hexdigest(),
            }
        cases[case.label] = {"prompt_tokens": len(case.expected_ids), "max_tokens": case.limit, "responses": results}
    return {"sources": sources, "hf_exact_token_and_finish_matches": token_matches,
            "hf_exact_text_matches": text_matches, "text_tail": text_tail, "cases": cases}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
