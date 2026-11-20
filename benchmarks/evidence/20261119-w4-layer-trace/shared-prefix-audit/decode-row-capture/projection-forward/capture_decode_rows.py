"""Exercise exact projection replacement in the real Riley HTTP decode path."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path

from capture_final_rows import targets, stem
from capture_reference import complete, same_completion
from replay_final_head import bf16
from trace_decode_layers import read_native

NATIVE_MANIFEST = "b5de696ebdda0025cdd4d3180baac4f134aa96e884714cc5c31eaea196554212"
REPLAY_MANIFEST = "bec6273c272a0af78e9b8e04cbe53b54b6d945c0abb38085ef035bf5f3599457"
ADDRESSABLE = 151_665
MODES = {
    "baseline": None,
    "identity-down": ("down_proj", "recorded", "down-override.bf16", ("mlp_down", "0")),
    "fp32-reduction": ("down_proj", "fixed-reduction", "down-override.bf16", ("mlp_down", "0")),
    "down-13-unsplit": ("down_proj", "heuristic-4", "down-override.bf16", ("mlp_down", "0")),
    "down-13-split64": ("down_proj", "heuristic-6", "down-override.bf16", ("mlp_down", "0")),
    "attention-31": ("o_proj", "heuristic-0", "o-proj-override.bf16", ("attention_projection", "0")),
    "baseline-repeat": None,
}


def digest(raw):
    return sha256(raw).hexdigest()


def prefix(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def measure(row, baseline, reference_id, actual_id):
    result = bf16(row[:ADDRESSABLE * 2])
    original = bf16(baseline[:ADDRESSABLE * 2])
    top = max(result)
    assert math.isfinite(top) and 0 <= actual_id < ADDRESSABLE and result[actual_id] == top
    return {
        "full_logits_sha256": digest(row),
        "changed_addressable_bf16_values": sum(a != b for a, b in zip(original, result, strict=True)),
        "hf_reference_bf16_score": result[reference_id],
        "selected_bf16_score": top,
        "hf_reference_rank": 1 + sum(score > result[reference_id] for score in result),
        "joint_maximum_count": sum(score == top for score in result),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("rows", type=Path)
    parser.add_argument("server_log", type=Path)
    args = parser.parse_args()
    evidence = args.rows.resolve().parent.parent.parent
    native = evidence / "w4-native-projection-plans-20261120.yvjrWvOj"
    replay = evidence / "w4-native-projection-replay-20261120.9XxkzP0C"
    assert digest((native / "result.sha256").read_bytes()) == NATIVE_MANIFEST
    assert digest((replay / "result.sha256").read_bytes()) == REPLAY_MANIFEST
    plans = json.loads((replay / "verified.json").read_bytes())["operators"]
    output = {}
    for _, (case, original, reference, offset, hf_id) in targets(args.fixtures).items():
        name = stem(original, offset)
        baseline_raw, original_rows = read_native(native / "riley/rows" / name, original, offset)
        baseline_id = original["tokens"][offset]
        modes = {}
        for mode, option in MODES.items():
            directory = args.rows / name
            directory.mkdir()
            selected = None
            if option is not None:
                operator, plan, file, point = option
                selected = (replay / operator / f"{name}.{plan}.bf16").read_bytes()
                assert len(selected) == 7168 and digest(selected) == plans[operator]["cases"][name][plan]["sha256"]
                (directory / file).write_bytes(selected)
            response = complete(args.address, args.model, case.prompt, diagnostic=True,
                                expected=original["prompt_tokens"], limit=case.limit)
            actual = response["tokens"]
            assert len(actual) > offset and actual[:offset] == original["tokens"][:offset]
            trace, rows = read_native(directory, original, offset)
            if mode in {"baseline", "identity-down", "baseline-repeat"}:
                assert trace == baseline_raw and same_completion(original, response) and original["usage"] == response["usage"]
            if option is not None:
                assert (rows[point] == selected)
                for key in rows:
                    if key == point:
                        break
                    assert rows[key] == original_rows[key], f"{name}/{mode}: preceding row {key} changed"
            recorded_ref = reference["hf_tokens"]
            modes[mode] = {
                "actual_token": actual[offset],
                "full_tokens_sha256": digest(json.dumps(actual).encode()),
                "text_sha256": digest(response["text"].encode()),
                "generated_tokens": len(actual),
                "finish_reason": response["finish"],
                "common_prefix_with_riley": prefix(actual, original["tokens"]),
                "common_prefix_with_hf": prefix(actual, recorded_ref),
                "same_completion_as_riley": same_completion(original, response),
                "trace_sha256": digest(trace),
                "applied_projection_sha256": digest(selected) if selected is not None else None,
                **measure(rows["logits", "-"], original_rows["logits", "-"], hf_id, actual[offset]),
            }
            directory.rename(args.rows / f"{name}.{mode}")
        output[name] = {
            "decode_offset": offset, "reference_hf_id": hf_id, "baseline_riley_id": baseline_id,
            "modes": modes,
        }
    assert len(output) == 2
    print(json.dumps({
        "original_native_manifest_sha256": NATIVE_MANIFEST,
        "candidate_plan_manifest_sha256": REPLAY_MANIFEST,
        "addressable_vocabulary": ADDRESSABLE,
        "mode_order": list(MODES),
        "cases": output,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
