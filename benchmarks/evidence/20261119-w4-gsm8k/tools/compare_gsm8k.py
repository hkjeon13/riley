"""Check two full runs and compare scores only when all input IDs match."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from gsm8k import FULL_QUESTIONS, protocol, scored, save


def read_run(path, verified_dataset):
    raw = path.read_bytes()
    document = json.loads(raw)
    rows = document["results"]
    if (document["status"] != "passed" or document["scope"] != "full"
            or document["questions"] != FULL_QUESTIONS or document["completed"] != FULL_QUESTIONS
            or document["errors"] != 0 or len(rows) != FULL_QUESTIONS
            or document["dataset_verified"] is not True or document["dataset"] != verified_dataset
            or document["protocol"] != protocol(True)):
        raise ValueError(f"incomplete or unbound GSM8K evidence in {path}")
    for index, row in enumerate(rows):
        expected = sha256(row["text"].encode()).hexdigest()
        if (type(row["index"]) is not int or row["index"] != index or "error" in row
                or type(row["correct"]) is not bool or row["text_sha256"] != expected
                or (row["predicted"], row["correct"]) != scored(row["text"], row["gold"])
                or not all(re.fullmatch("[0-9a-f]{64}", row[key])
                           for key in ("request_sha256", "prompt_sha256", "prompt_ids_sha256", "output_ids_sha256"))):
            raise ValueError(f"invalid raw result at question {index} in {path}")
    correct = sum(row["correct"] for row in rows)
    if document["correct"] != correct or document["accuracy"] != correct / len(rows):
        raise ValueError(f"incorrect claimed summary in {path}")
    return document, sha256(raw).hexdigest()


def compare(dataset_path, baseline_path, candidate_path):
    verified_dataset = json.loads(dataset_path.read_bytes())["files"]
    baseline, baseline_hash = read_run(baseline_path, verified_dataset)
    candidate, candidate_hash = read_run(candidate_path, verified_dataset)
    ids_differ = []
    riley_only = vllm_only = text_matches = token_matches = 0
    for ref, got in zip(baseline["results"], candidate["results"], strict=True):
        if any(ref[key] != got[key] for key in ("request_sha256", "prompt_sha256", "gold")):
            raise ValueError(f"engines evaluated different questions or request settings at {ref['index']}")
        if ref["prompt_ids_sha256"] != got["prompt_ids_sha256"]:
            ids_differ.append(ref["index"])
        text_matches += int(ref["text_sha256"] == got["text_sha256"])
        token_matches += int(ref["output_ids_sha256"] == got["output_ids_sha256"])
        riley_only += int(got["correct"] and not ref["correct"])
        vllm_only += int(ref["correct"] and not got["correct"])
    gap = (candidate["correct"] - baseline["correct"]) * 100 / FULL_QUESTIONS
    status = "unmatched-prompt-ids" if ids_differ else "passed" if gap >= -1.2 else "failed"
    return {
        "status": status, "questions": FULL_QUESTIONS,
        "riley_correct": candidate["correct"], "vllm_correct": baseline["correct"],
        "riley_only_correct": riley_only, "vllm_only_correct": vllm_only,
        "matching_text": text_matches, "matching_output_ids": token_matches,
        "unmatched_prompt_ids": ids_differ,
        "gap_percentage_points": gap if not ids_differ else None,
        "riley_sha256": candidate_hash, "vllm_sha256": baseline_hash,
        "dataset": verified_dataset,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "vllm", "riley", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = compare(args.dataset, args.vllm, args.riley)
    save(args.output, result)
    print(result["status"], result["gap_percentage_points"])
    raise SystemExit(int(result["status"] != "passed"))
