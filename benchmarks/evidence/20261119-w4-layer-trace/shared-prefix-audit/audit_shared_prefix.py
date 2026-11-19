"""Locate W4 first divergences and near ties on the engines' shared prefixes."""

from argparse import ArgumentParser
from collections import Counter
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path

from long_accuracy import common_prefix, read_capture, read_workload, riley_margin, same_completion, terminal_normalized


THRESHOLD = 0.25
SUITES = ("long", "holdout")
FILES = ("riley/capture.json.gz", "hf/reference.json", "vllm/results.json.gz")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def checked_files(directory, suite):
    entries = [line.split(maxsplit=1) for line in (directory / "SHA256SUMS").read_text().splitlines()]
    checks_by_path = {path.removeprefix("./"): digest for digest, path in entries}
    check(len(checks_by_path) == len(entries), f"{suite}: repeated checksum path")
    names = (*FILES, "holdout.json.gz") if suite == "holdout" else FILES
    actual = {name: sha256((directory / name).read_bytes()).hexdigest() for name in names}
    check(all(value == checks_by_path[name] for name, value in actual.items()), f"{suite}: capture digest differs")
    return actual


def score_check(scores, first, second):
    top = scores["top"]
    check(len(top) == 5 and all(math.isfinite(row["logit"]) for row in top), "invalid HF top-five scores")
    check(all(a["logit"] >= b["logit"] for a, b in zip(top, top[1:]))
          and scores["margin"] == top[0]["logit"] - top[1]["logit"], "HF margin differs from the stored scores")
    result = {name: scores["selected_scores"][str(token)] for name, token in (("riley", first), ("hf", second))}
    check(all(math.isfinite(value) and value <= top[0]["logit"] for value in result.values()),
          "invalid stored score for a divergent token")
    winner = top[0]["id"] if scores["margin"] > 0 else None
    return {"scores": result, "hf_id_minus_riley_id": result["hf"] - result["riley"],
            "top_pair_gap": scores["margin"], "unique_winner": winner}


def audit_suite(directory, suite):
    hashes = checked_files(directory, suite)
    holdout = directory / "holdout.json.gz" if suite == "holdout" else None
    workload = read_workload(holdout)
    captured = read_capture(directory / FILES[0], workload)
    hf = json.loads((directory / FILES[1]).read_bytes())
    vllm = json.loads(gzip.decompress((directory / FILES[2]).read_bytes()))
    check(hf.get("workload_sha256") == vllm.get("workload_sha256") == workload.sha256,
          f"{suite}: workload binding differs")
    eos = set(hf["eos_token_ids"])
    rows = []
    cases = zip(workload.cases, captured["baseline"], captured["diagnostic"], hf["cases"], vllm["reference"], strict=True)
    for index, (case, base, diagnostic, ref, other) in enumerate(cases):
        check(ref["case"] == index and ref.get("label", case.label) == case.label and ref["hf_tokenizer_ids_match"]
              and len(base["prompt_tokens"]) == ref["prompt_tokens"]
              and base["prompt_tokens"] == other["prompt_tokens"]
              and same_completion(base, diagnostic), f"{suite}/{index}: prompt or diagnostic bindings differ")
        first = terminal_normalized(base["tokens"], base["finish"], eos)
        second = terminal_normalized(ref["hf_tokens"], ref["hf_finish"], eos)
        third = terminal_normalized(other["tokens"], other["finish"], eos)
        prefix = common_prefix(first, second)
        check(prefix == ref["common_prefix"] and ref["exact_token_match"] == (first == second),
              f"{suite}/{index}: recorded first difference differs from the tokens")
        logprobs = diagnostic["logprobs"]
        check(all(len(logprobs[field]) == len(base["tokens"]) for field in ("tokens", "token_logprobs", "top_logprobs")),
              f"{suite}/{index}: public logprob rows are incomplete")
        margins = Counter()
        for step in range(prefix):
            gap = riley_margin(base, diagnostic, step)
            margins["shared_decisions"] += 1
            if gap is not None:
                margins["reported_top_two_available"] += 1
                margins["reported_tie"] += gap == 0
                margins["reported_gap_at_most_0_25"] += gap <= THRESHOLD
        row = {"case": index, "label": case.label, "prompt_tokens": ref["prompt_tokens"], "shared_prefix": dict(margins)}
        divergence = ref["first_divergence"]
        if first == second:
            check(divergence is None, f"{suite}/{index}: unexpected first-divergence record")
        else:
            check(prefix < min(len(first), len(second)) and divergence["offset"] == prefix,
                  f"{suite}/{index}: decision-point comparison requires two divergent token IDs")
            ids = first[prefix], second[prefix]
            check(ids == (divergence["riley_id"], divergence["hf_id"]), "divergent IDs differ")
            gap = riley_margin(base, diagnostic, prefix)
            check(gap is not None and gap == divergence["riley_reported_top_logprob_gap"], "Riley recorded gap differs")
            generation = score_check(divergence["hf_generation_scores"], *ids)
            sdpa = score_check(divergence["hf_sdpa_replay"], *ids)
            check(generation == sdpa and sdpa["unique_winner"] in (None, ids[1])
                  and sdpa["scores"]["hf"] == divergence["hf_sdpa_replay"]["top"][0]["logit"],
                  "HF SDPA shared-prefix replay differs")
            eager = score_check(divergence["hf_eager_replay"], *ids)
            if third[:prefix] != first[:prefix]:
                vllm_choice = {"state": "already_diverged_or_ended", "riley_common_prefix": common_prefix(first, third)}
            else:
                token = third[prefix] if prefix < len(third) else None
                outcome = "riley" if token == ids[0] else "hf_sdpa" if token == ids[1] else "third" if token is not None else "ended"
                vllm_choice = {"state": outcome, "id": token}
            row["divergence"] = {"offset": prefix, "stage": "prefill" if prefix == 0 else "decode",
                                 "riley_id": ids[0], "hf_sdpa_id": ids[1], "riley_reported_gap": gap,
                                 "hf_sdpa": sdpa, "hf_eager": eager, "vllm_at_same_prefix": vllm_choice}
        rows.append(row)
    return {"source_sha256": hashes, "cases": rows}


def audit(root):
    suites = {name: audit_suite(root / f"20261119-w4-{name}", name) for name in SUITES}
    cases = [row for suite in suites.values() for row in suite["cases"]]
    mismatches = [row["divergence"] for row in cases if "divergence" in row]
    shared = Counter()
    for row in cases:
        shared.update(row["shared_prefix"])
    matched_case_ties = sum(row["shared_prefix"].get("reported_tie", 0) > 0
                            for row in cases if "divergence" not in row)
    eager_states = Counter("riley" if row["hf_eager"]["unique_winner"] == row["riley_id"] else
                           "hf_sdpa" if row["hf_eager"]["unique_winner"] == row["hf_sdpa_id"] else
                           "top_tie" if row["hf_eager"]["unique_winner"] is None else "third"
                           for row in mismatches)
    return {"scope": "Existing Qwen2.5-7B-Instruct results, same normalized prefix and no new inference. "
                     "Riley top logprobs are keyed by rendered text, so the reported gap can omit tied token IDs.",
            "riley_gap_threshold": THRESHOLD, "suites": suites,
            "summary": {"total_inputs": len(cases), "exact_complete_generation": len(cases) - len(mismatches),
                        "matched_cases_with_a_reported_tie": matched_case_ties,
                        "shared_prefix": dict(shared), "divergences": len(mismatches),
                        "divergences_by_stage": dict(Counter(row["stage"] for row in mismatches)),
                        "first_divergence_riley_gap_at_most_0_25": sum(row["riley_reported_gap"] <= THRESHOLD for row in mismatches),
                        "first_divergence_riley_reported_ties": sum(row["riley_reported_gap"] == 0 for row in mismatches),
                        "first_divergence_max_hf_sdpa_pair_gap": max(row["hf_sdpa"]["hf_id_minus_riley_id"] for row in mismatches),
                        "eager_unique_winners_or_top_ties": dict(eager_states),
                        "vllm_at_same_prefix": dict(Counter(row["vllm_at_same_prefix"]["state"] for row in mismatches))}}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.evidence_root), indent=2, allow_nan=False))
