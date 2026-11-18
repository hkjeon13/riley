"""Capture pinned first-token cases for one Riley attention or KV-write arm."""

import argparse
from pathlib import Path

from layer_trace import pinned_case
from long_accuracy import complete, same_completion, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("fixture", type=Path)
    parser.add_argument("mode", choices=("baseline", "shared-kv", "separate-write"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = {}
    for case in (3, 9):
        row, prompt = pinned_case(args.fixture, case)
        ordinary = complete(args.address, args.model, prompt, expected=row["prompt_tokens"])
        diagnostic = complete(args.address, args.model, prompt, diagnostic=True, expected=row["prompt_tokens"])
        repeated = complete(args.address, args.model, prompt, expected=row["prompt_tokens"])
        pinned_output = same_completion(ordinary, row)
        if not same_completion(ordinary, repeated) or not same_completion(ordinary, diagnostic):
            raise ValueError(f"case {case} changed between ordinary, diagnostic, or repeated requests")
        if args.mode == "baseline" and not pinned_output:
            raise ValueError(f"the baseline failed to match the pinned case {case}")
        cases[str(case)] = {"prompt_sha256": row["prompt_sha256"], "historical_first_id": row["tokens"][0],
                            "matches_original_ids_text_finish": pinned_output, "response": diagnostic}
    save(args.output, {"mode": args.mode, "cases": cases})


if __name__ == "__main__":
    main()
