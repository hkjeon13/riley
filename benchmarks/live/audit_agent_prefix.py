"""Audit service-eligible prompt block reuse for a recorded synthetic workflow run."""

import argparse
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


def audit(records, model, executable, tokenizer, *, same_workflow=False):
    requests, metadata = [], []
    for work in records:
        tools, prompt, docs = fixture(work["seed"], work["document_pairs"])
        messages = history_messages(work["seed"], work.get("history_entries", 0)) + [
            {"role": "user", "content": prompt}
        ]
        if fingerprint([prompt, docs]) != work["fixture_sha256"]:
            raise ValueError("fixture does not match recording")
        for index, row in enumerate(work["turns"]):
            request = turn_request(
                model, messages, tools, ("list_reports", "read_reports", None)[index]
            )
            local_step = row.get("dispatch") == "application"
            hash_field = "planned_request_sha256" if local_step else "input_sha256"
            if fingerprint(request) != row[hash_field]:
                raise ValueError("reconstructed request differs from submitted input")
            if not local_step:
                requests.append(json.dumps(request))
                metadata.append((work["workflow_id"], row))
            if "call" in row:
                append_tool_result(messages, [row["call"]], docs, index)
    result = subprocess.run(
        [executable, tokenizer],
        input="\n".join(requests) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    encoded = [json.loads(line) for line in result.stdout.splitlines()]
    receipt = []
    for (workflow, row), data in zip(metadata, encoded, strict=True):
        ids = data["ids"]
        if len(ids) != row["usage"]["prompt_tokens"]:
            raise ValueError("native tokenizer and served prompt token counts differ")
        # Finished prior calls have committed every complete prompt block.
        # Other in-flight calls may have published more by the time of adoption.
        prior = [
            other
            for (other_workflow, t), other in zip(metadata, encoded, strict=True)
            if t["done_ns"] <= row["sent_ns"]
            and (not same_workflow or workflow == other_workflow)
        ]
        shared = max(
            (
                next(
                    (i for i, (a, b) in enumerate(zip(ids, other["ids"])) if a != b),
                    min(len(ids), len(other["ids"])),
                )
                for other in prior
            ),
            default=0,
        )
        eligible = min(shared, len(ids) - 1) // 16 * 16
        receipt.append(
            {
                "workflow": workflow,
                "turn": row["turn"],
                "request_id": row["response_id"],
                "total_tokens": len(ids),
                "prior_completed_eligible_tokens": eligible,
                "actual_hit_tokens": (
                    row["usage"].get("prompt_tokens_details") or {}
                ).get("cached_tokens", 0),
                "offline_normalize_us": round(data["normalize_ns"] / 1000, 1),
                "offline_encode_us": round(data["encode_ns"] / 1000, 1),
            }
        )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflows", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("--model", default="m")
    parser.add_argument(
        "--same-workflow",
        action="store_true",
        help="consider only earlier completed prompts in the same workflow",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                json.loads(args.workflows.read_text()),
                args.model,
                args.executable,
                args.tokenizer,
                same_workflow=args.same_workflow,
            ),
            indent=2,
        )
    )
