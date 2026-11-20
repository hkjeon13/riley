"""Replay the exact HF separate-projection row in an otherwise unchanged Riley decode."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from capture_final_rows import targets, stem
from capture_reference import complete, require, same_completion
from replay_final_head import bf16
from trace_decode_layers import read_native


REFERENCE_SHA256 = "9bb140a4290330432613cd1a861bd629620b085be021a3fcfabd0e2e48fd9491"
PRIOR_FORWARD_SHA256 = "9a8dddc773fc7718d4ba5f01980456b518ffdc126ad700acc7c6342097a623ef"
PREBIAS = ("qkv_prebias", "0")
LOGITS = ("logits", "-")
MODES = ("baseline", "hf-separate", "baseline-repeat")


def digest(raw):
    return sha256(raw).hexdigest()


def report(root, expected):
    raw = (root / "verified.json").read_bytes()
    require(digest(raw) == expected, "previous GPU report differs")
    return json.loads(raw)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("rows", type=Path)
    parser.add_argument("server_log", type=Path)
    args = parser.parse_args()
    evidence = args.rows.parent.parent.parent
    crossover = evidence / "w4-projection-crossover-20261120.J9REWUkf"
    previous = evidence / "w4-qkv-forward-retry-20261120.QhDayeS4"
    reference_cases = report(crossover, REFERENCE_SHA256)["cases"]
    previous_cases = report(previous, PRIOR_FORWARD_SHA256)["cases"]
    cases = {}
    for _, (case, original, _, offset, reference_id) in targets(args.fixtures).items():
        name = stem(original, offset)
        hf = (crossover / "operands" / f"{name}.hf.bf16").read_bytes()
        expected = reference_cases[name]["replays"]["torch-separate"]["sha256"]
        require(len(hf) == 9216 and digest(hf) == expected, "exact HF projection data differs")
        modes = {}
        for mode in MODES:
            folder = args.rows / name
            folder.mkdir()
            if mode == "hf-separate":
                (folder / "qkv-override.bf16").write_bytes(hf)
            response = complete(args.address, args.model, case.prompt, diagnostic=True,
                                expected=original["prompt_tokens"], limit=case.limit)
            tokens = response["tokens"]
            require(len(tokens) > offset and tokens[:offset] == original["tokens"][:offset],
                    "input history before the replay differs")
            raw, rows = read_native(folder, original, offset)
            modes[mode] = (response, raw, rows)
            folder.rename(args.rows / f"{name}.{mode}")
        response, base_raw, base_rows = modes["baseline"]
        require(digest(base_raw) == previous_cases[name]["trace_sha256"]["baseline"],
                "current baseline differs from the prior served trace")
        require(modes["baseline-repeat"][1] == base_raw and all(
            same_completion(original, modes[mode][0]) for mode in ("baseline", "baseline-repeat")),
            "unchanged controls disagree")
        changed_response, changed_raw, changed_rows = modes["hf-separate"]
        require(changed_rows[PREBIAS] == hf and changed_rows[PREBIAS] != base_rows[PREBIAS],
                "HF values were not used by the model")
        require(changed_rows["input_norm", "0"] == base_rows["input_norm", "0"]
                and changed_rows["embedding", "-"] == base_rows["embedding", "-"],
                "the intervention changed upstream input")
        baseline_id, actual_id = response["tokens"][offset], changed_response["tokens"][offset]
        original_logits, replay_logits = (bf16(rows[LOGITS][:151665 * 2]) for rows in (base_rows, changed_rows))
        require(original_logits[baseline_id] == max(original_logits)
                and replay_logits[actual_id] == max(replay_logits), "generated ID differs from the saved top logit")
        cases[name] = {
            "selected_token": {"baseline": baseline_id, "hf_replay": actual_id, "hf_reference": reference_id},
            "same_full_completion": same_completion(original, changed_response),
            "reference_projection_sha256": digest(hf),
            "previous_plan_31_exact_trace": digest(changed_raw) == previous_cases[name]["trace_sha256"]["rank-2"],
            "selected_minus_reference_logit": [original_logits[baseline_id] - original_logits[reference_id],
                                               replay_logits[actual_id] - replay_logits[reference_id]],
            "hf_ranks": [1 + sum(score > logits[reference_id] for score in logits)
                         for logits in (original_logits, replay_logits)],
            "trace_sha256": {mode: digest(run[1]) for mode, run in modes.items()},
        }
    require(len(cases) == 2, "two pinned cases are required")
    print(json.dumps({"mode_order": MODES, "reference_sha256": REFERENCE_SHA256,
                      "prior_forward_sha256": PRIOR_FORWARD_SHA256, "cases": cases}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
