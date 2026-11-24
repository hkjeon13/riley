"""Evaluate an immutable prompt-lookup rule using only each causal prefix to propose."""

import argparse
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "live"))
from agent_workflow import fingerprint
from fixed_agent_decode import freeze


def replay(prompt, output):
    positions = defaultdict(list)
    history = []

    def observe(token):
        history.append(token)
        for n in range(2, min(6, len(history)) + 1):
            positions[tuple(history[-n:])].append(len(history))

    for token in prompt:
        observe(token)
    offset = attempts = accepted = proposed = passes = 0
    while offset < len(output):
        draft = []
        for n in range(min(6, len(history)), 1, -1):
            earlier = positions[tuple(history[-n:])]
            if len(earlier) > 1:
                start = earlier[-2]
                draft = history[start : start + 4]
                break
        # References are consulted only after the proposer has returned.
        reference = output[offset : offset + len(draft)]
        agreement = sum(
            1
            for _ in itertools.takewhile(
                lambda pair: pair[0] == pair[1], zip(draft, reference)
            )
        )
        produced = min(1 + agreement, len(output) - offset)
        for t in output[offset : offset + produced]:
            observe(t)
        attempts += bool(draft)
        accepted += agreement
        proposed += len(draft)
        passes += 1
        offset += produced
    return {
        "emitted": len(output),
        "target_passes": passes,
        "passes_with_proposal": attempts,
        "tokens_proposed": proposed,
        "tokens_matching_reference": accepted,
        "optimistic_pass_reduction": 1 - passes / len(output),
    }


def read(p):
    return json.loads(p.read_text())


def fixed(args):
    inputs = []
    trials = [t for t in read(args.launch / "fixed.json") if t["phase"] == "identity"]
    for t in trials:
        work = next(
            r
            for r in read(args.source / f"workflows-{t['history']}.json")
            if r["workflow_id"] == t["result"]["name"]
        )
        inputs.append(freeze(work, "m"))
    result = subprocess.run(
        [args.binary, args.tokenizer],
        input="".join(json.dumps(req) + "\n" for req in inputs),
        text=True,
        capture_output=True,
        check=True,
    )
    cases = []
    for row, line in zip(trials, result.stdout.splitlines(), strict=True):
        m = row["result"]
        ids = json.loads(line)["ids"]
        if fingerprint(ids) != m["prompt_ids_sha256"] or len(m["generated_ids"]) != 64:
            raise ValueError("offline prompt or original target output differs")
        cases.append(
            {
                "history": row["history"],
                "stream": m["name"],
                "input_sha256": m["input_sha256"],
                **replay(ids, m["generated_ids"]),
            }
        )
    return cases


def natural(args):
    rows = read(args.cases)
    reg = read(Path(__file__).with_name("lookup-decode-registration.json"))
    if [
        [
            r["workflow"]["seed"],
            r["workflow"]["history_entries"],
            r["workflow"]["document_pairs"],
        ]
        for r in rows
    ] != reg["heldout_cases"]:
        raise ValueError("heldout workload differs")
    submitted = [r for r in rows if r["final_request"] is not None]
    raw = subprocess.run(
        [args.binary, args.tokenizer],
        input="".join(json.dumps(r["final_request"]) + "\n" for r in submitted),
        text=True,
        capture_output=True,
        check=True,
    )
    encoded = iter(raw.stdout.splitlines())
    cases = []
    for r in rows:
        w = r["workflow"]
        summary = {
            "seed": w["seed"],
            "history": w["history_entries"],
            "pairs": w["document_pairs"],
            "workflow_ok": w["ok"],
            "workflow_error": w.get("error"),
        }
        if r["final_request"] is not None:
            m = w["turns"][-1]
            ids = json.loads(next(encoded))["ids"]
            if (
                fingerprint(ids) != m["prompt_ids_sha256"]
                or fingerprint(r["final_request"]) != m["input_sha256"]
            ):
                raise ValueError("heldout causal prompt differs from request")
            output = m["generated_ids"]
            count = m["usage"]["completion_tokens"]
            if (
                len(output) not in (count, count - 1)
                or len(output) < 2
                or not m.get("done_ns")
            ):
                raise ValueError("incomplete target output")
            reduction = replay(ids + output[:1], output[1:])
            remaining_ms = (m["content_ns"][-1] - m["content_ns"][0]) / 1e6
            summary.update(
                reported_completion_tokens=count,
                returned_ids=len(output),
                final_decode_ms=remaining_ms,
                uniform_cost_potential_ms=remaining_ms
                * reduction["optimistic_pass_reduction"],
                **reduction,
            )
        cases.append(summary)
    if next(encoded, None) is not None:
        raise ValueError("unexpected extra tokenizations")
    return cases


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    baseline = sub.add_parser("fixed")
    baseline.add_argument("launch", type=Path)
    baseline.add_argument("source", type=Path)
    holdout = sub.add_parser("natural")
    holdout.add_argument("cases", type=Path)
    for child in (baseline, holdout):
        child.add_argument("binary", type=Path)
        child.add_argument("tokenizer", type=Path)
    args = p.parse_args()
    print(json.dumps((fixed if args.command == "fixed" else natural)(args), indent=2))
