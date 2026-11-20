"""Measure the model response after replaying one verified projection row."""

from argparse import ArgumentParser
import json
import math
from pathlib import Path

from capture_final_rows import targets, stem
from capture_reference import complete, require, same_completion
from replay_final_head import bf16
from trace_decode_layers import difference, digest, read_native


TRACE_DIGEST = "b7c459a5558d1d371228c751b83ad6bf29aeed4ff6100f3d472300bf59ece10a"
PLAN_DIGEST = "339bed90fe18cf99cbce5cd91fcc7b9b1d1e4eb9e69cd50d83a4defbe49c83a3"
MODES = ("baseline", "recorded", "rank-2", "baseline-repeat")
PREBIAS = ("qkv_prebias", "0")
LOGITS = ("logits", "-")


def checked_report(folder, sha):
    data = (folder / "verified.json").read_bytes()
    require(digest(data) == sha, "recorded W4 evidence differs")
    return json.loads(data)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("rows", type=Path)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    stage = args.rows.parent.parent
    earlier = stage.parent / "w4-decode-layer0-ops-20261120.hYAOtc0n"
    operator = stage.parent / "w4-native-plan-controls-20261120.n1qiR9dQ"
    known_traces = checked_report(earlier, TRACE_DIGEST)["cases"]
    plans = checked_report(operator, PLAN_DIGEST)["plans"]
    require(plans["recorded"]["config"]["id"] == 21
            and plans["rank-2"]["config"] == {**plans["recorded"]["config"], "id": 31},
            "the alternative must differ only by native algorithm ID")
    completed = {}
    for _, (case, original, _, offset, other) in targets(args.fixtures).items():
        name = stem(original, offset)
        runs = {}
        for mode in MODES:
            row = args.rows / name
            row.mkdir()
            if mode in plans:
                content = (operator / "operands" / f"{name}.{mode}.bf16").read_bytes()
                require(len(content) == 9216 and digest(content) == plans[mode]["cases"][name]["sha256"],
                        "QKV replay data differs from the verified operator trial")
                (row / "qkv-override.bf16").write_bytes(content)
            response = complete(args.address, args.model, case.prompt, diagnostic=True,
                                expected=original["prompt_tokens"], limit=case.limit)
            served = response["tokens"]
            require(len(served) > offset and served[:offset] == original["tokens"][:offset],
                    f"{name}/{mode}: response before the intervention differs")
            raw, values = read_native(row, original, offset)
            runs[mode] = {"response": response, "raw": raw, "values": values}
            row.rename(args.rows / f"{name}.{mode}")
        baseline = runs["baseline"]
        require(digest(baseline["raw"]) == known_traces[name]["source_trace_sha256"]["riley"],
                "isolated baseline differs from the previous native serving trace")
        require(digest(baseline["values"][PREBIAS]) == plans["recorded"]["cases"][name]["sha256"],
                "baseline does not implement the recorded plan's output")
        for mode in ("baseline", "recorded", "baseline-repeat"):
            control = runs[mode]
            require(control["raw"] == baseline["raw"] and same_completion(original, control["response"])
                    and original["usage"] == control["response"]["usage"],
                    f"{name}/{mode}: exact reference or sham response changed")
        replay = runs["rank-2"]
        require(digest(replay["values"][PREBIAS]) == plans["rank-2"]["cases"][name]["sha256"]
                and replay["values"][PREBIAS] != baseline["values"][PREBIAS],
                f"{name}: the alternate BF16 projection did not reach the split")
        for key in ("embedding", "-",), ("input_norm", "0"):
            require(replay["values"][key] == baseline["values"][key], f"{name}: replay changed upstream input")
        base_token = original["tokens"][offset]
        chosen = replay["response"]["tokens"][offset]
        scores = {}
        for mode, run in runs.items():
            logits = run["values"][LOGITS]
            scores[mode] = {str(token): bf16(logits[token * 2:token * 2 + 2])[0]
                            for token in dict.fromkeys((base_token, other, chosen))}
            require(all(math.isfinite(score) for score in scores[mode].values()), f"{name}: nonfinite logit")
        completed[name] = {
            "selected_token": {"baseline": base_token, "replay": chosen, "hf_reference": other},
            "same_full_completion": same_completion(original, replay["response"]),
            "scores": scores,
            "row_differences": [
                {"stage": key[0], "layer": key[1], **difference(baseline["values"][key], replay["values"][key])}
                for key in baseline["values"]
            ],
            "trace_sha256": {mode: digest(run["raw"]) for mode, run in runs.items()},
        }
    require(len(completed) == 2, "exactly two previously verified C1 rows are required")
    print(json.dumps({"mode_order": MODES, "baseline_and_sham_bit_equal": True,
                      "controls_report_sha256": PLAN_DIGEST, "cases": completed}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
