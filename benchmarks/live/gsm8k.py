#!/usr/bin/env python3
"""Evaluate GSM8K with the historical four-shot prompt and last-number scoring.

The data directory contains the JSONL train and test files from openai/gsm8k
main. A full run requires all 1,319 test questions. A supplied dataset manifest
binds both files before any request is sent.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import urllib.error
import urllib.request

from long_accuracy import save, token_ids

FULL_QUESTIONS = 1319
SHOTS = 4
MAX_TOKENS = 256
STOP = "Question:"


def protocol(require_ids):
    return {"shots": SHOTS, "max_tokens": MAX_TOKENS, "temperature": 0, "stop": [STOP],
            "concurrency": 32, "scoring": "historical-last-number", "return_token_ids": require_ids}


def gold_answer(row):
    return row["answer"].split("####")[-1].strip().replace(",", "")


def number(text):
    found = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return found[-1].replace(",", "").rstrip(".") if found else None


def scored(text, gold):
    predicted = number(text.split("####")[-1] if "####" in text else text)
    try:
        correct = predicted is not None and float(predicted) == float(gold)
    except ValueError:
        correct = False
    return predicted, correct


def load_data(directory, manifest_path=None):
    splits = {}
    metadata = {}
    for split in ("train", "test"):
        raw = (directory / f"{split}.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        for row in rows:
            if (not isinstance(row["question"], str) or not row["question"].strip()
                    or not isinstance(row["answer"], str) or "####" not in row["answer"]):
                raise ValueError(f"invalid GSM8K {split} row")
            if not math.isfinite(float(gold_answer(row))):
                raise ValueError(f"invalid GSM8K {split} answer")
        splits[split] = rows
        metadata[split] = {"sha256": sha256(raw).hexdigest(), "rows": len(rows)}
    if len(splits["test"]) != FULL_QUESTIONS or len(splits["train"]) < SHOTS:
        raise ValueError("expected all 1,319 test questions and four train examples")
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_bytes())
        if (manifest.get("dataset") != "openai/gsm8k:main" or manifest.get("files") != metadata):
            raise ValueError("GSM8K dataset digest or row count differs from the reviewed files")
    return splits, metadata


def make_prompt(shots, question):
    return shots + f"Question: {question}\nAnswer:"


def id_digest(ids):
    return sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def inspect_completion(reply, require_ids):
    row, = reply["choices"]
    text, finish = row["text"], row["finish_reason"]
    usage = reply["usage"]
    if (not isinstance(text, str) or finish not in ("stop", "length")
            or type(usage["prompt_tokens"]) is not int or usage["prompt_tokens"] <= 0
            or type(usage["completion_tokens"]) is not int or not 0 <= usage["completion_tokens"] <= MAX_TOKENS
            or (finish == "length" and usage["completion_tokens"] != MAX_TOKENS)):
        raise ValueError("invalid text, finish reason or token usage")
    result = {"text": text, "text_sha256": sha256(text.encode()).hexdigest(), "finish_reason": finish,
              "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"]}
    if require_ids:
        raw_prompt_ids = row.get("prompt_token_ids")
        prompt_ids = token_ids(raw_prompt_ids if raw_prompt_ids is not None else reply.get("prompt_token_ids"))
        generated = token_ids(row["token_ids"])
        if len(prompt_ids) != usage["prompt_tokens"] or len(generated) > MAX_TOKENS:
            raise ValueError("incoherent returned token IDs")
        result.update(prompt_ids_sha256=id_digest(prompt_ids), output_ids_sha256=id_digest(generated),
                      returned_output_ids=len(generated))
    return result


def ask(address, model, shots, index, row, require_ids):
    prompt = make_prompt(shots, row["question"])
    body = {"model": model, "prompt": prompt, "max_tokens": MAX_TOKENS, "temperature": 0, "stop": [STOP]}
    if require_ids:
        body["return_token_ids"] = True
    encoded = json.dumps(body).encode()
    record = {"index": index, "prompt_sha256": sha256(prompt.encode()).hexdigest(),
              "request_sha256": sha256(encoded).hexdigest(),
              "gold": gold_answer(row)}
    request = urllib.request.Request(f"http://{address}/v1/completions", encoded,
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            reply = json.load(response)
        record.update(inspect_completion(reply, require_ids))
        record["predicted"], record["correct"] = scored(record["text"], record["gold"])
    except urllib.error.HTTPError as error:
        record["error"] = f"HTTP {error.code}"
    except (OSError, ValueError, KeyError, TypeError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def evaluate(args):
    splits, metadata = load_data(args.data, args.dataset_manifest)
    shots = "".join(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n" for row in splits["train"][:SHOTS])
    sample = splits["test"][:args.limit] if args.limit else splits["test"]

    def run(task):
        index, row = task
        return ask(args.address, args.model, shots, index, row, args.require_token_ids)

    with ThreadPoolExecutor(32) as pool:
        results = []
        for row in pool.map(run, enumerate(sample)):
            results.append(row)
            if len(results) % 128 == 0:
                print(f"completed {len(results)}/{len(sample)} attempts", flush=True)
    errors = sum("error" in row for row in results)
    correct = sum(row.get("correct") is True for row in results)
    return {"status": "failed" if errors else "passed", "scope": "sample" if args.limit else "full",
            "accuracy": None if errors else correct / len(results),
            "questions": len(results), "completed": len(results) - errors, "errors": errors,
            "correct": correct, "dataset": metadata, "dataset_verified": args.dataset_manifest is not None,
            "protocol": protocol(args.require_token_ids),
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--require-token-ids", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.limit <= FULL_QUESTIONS:
        parser.error(f"limit must be between zero and {FULL_QUESTIONS}")
    if args.out.exists():
        parser.error("output already exists")
    result = evaluate(args)
    save(args.out, result)
    print(json.dumps({key: result[key] for key in ("status", "scope", "accuracy", "questions",
                                                 "correct", "errors")}))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
