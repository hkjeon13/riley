"""Score two output tokens after replacing saved sublayer updates in a frozen replay."""

from argparse import ArgumentParser
import gzip
import json
import math
from pathlib import Path

from replay_final_head import TOKEN_IDS, bf16, digest, read_arms, require, round_bf16
from replay_residual_margin import dot, read_rows


ENGINES = ("riley", "hf")
SUBLAYERS = (("attention_projection", "attention_residual"), ("mlp_down", "layer_residual"))
TARGET_LAYERS = (21, 22, 27)


def bind_weights(weights, width, metadata):
    raw = weights.read_bytes()
    require(len(raw) == 6 * width, "expected exactly two head rows and one final norm row")
    chunks = [raw[index:index + 2 * width] for index in range(0, len(raw), 2 * width)]
    checksums = [metadata["weights"]["tokens"][str(token)] for token in TOKEN_IDS]
    checksums.append(metadata["norm_weight"]["weight_sha256"])
    require([digest(chunk) for chunk in chunks] == checksums, "weights differ from the trace receipt")
    a, b, gamma = map(bf16, chunks)
    return (a, b), gamma


def rows_in_order(capture, layers):
    initial = capture["embedding", "-"]
    updates = []
    states = [initial]
    for layer in range(layers):
        for change, after in SUBLAYERS:
            updates.append(capture[change, str(layer)])
            states.append(capture[after, str(layer)])
    return updates, states


def bf16_residual_add(base, delta):
    return [round_bf16(x + y) for x, y in zip(base, delta, strict=True)]


def frozen_replacement(updates, observed_states, other_updates, selection):
    first = min(selection)
    result = observed_states[first]
    for index in range(first, len(updates)):
        delta = other_updates[index] if index in selection else updates[index]
        result = bf16_residual_add(result, delta)
    return result


def normalized(row, gamma, epsilon):
    inverse_rms = 1 / math.sqrt(dot(row, row) / len(row) + epsilon)
    return [round_bf16(round_bf16(value * inverse_rms) * weight)
            for value, weight in zip(row, gamma, strict=True)]


def score(norm, head_rows):
    full = [dot(norm, row) for row in head_rows]
    rounded = list(map(round_bf16, full))
    winner = None
    if rounded[0] != rounded[1]:
        winner = TOKEN_IDS[int(rounded[1] > rounded[0])]
    return {
        "fp64_margin_5257_minus_264": full[1] - full[0],
        "bf16_scores": dict(zip(map(str, TOKEN_IDS), rounded, strict=True)),
        "pair_winner_token_id": winner,
    }


def replay_case(trace, fixture, case, shape, metadata, head_rows, gamma):
    width, vocabulary, layers, epsilon = shape
    norms, identities = read_arms(trace, fixture, case, width, vocabulary, metadata["checkpoint_manifest_sha256"])
    captures = read_rows(trace, case, layers, width)
    require(captures[0]["embedding", "-"] == captures[1]["embedding", "-"], "different traced embedding")
    ordered = dict(zip(ENGINES, (rows_in_order(capture, layers) for capture in captures), strict=True))
    output = {}
    for index, name in enumerate(ENGINES):
        updates, observed_states = ordered[name]
        other_updates, other_states = ordered[ENGINES[1 - index]]
        for base, delta, result in zip(observed_states[:-1], updates, observed_states[1:], strict=True):
            require(bf16_residual_add(base, delta) == result, "a recorded BF16 residual addition differed")
        baseline_norm = normalized(observed_states[-1], gamma, epsilon)
        require(baseline_norm == norms[name], "final norm did not match the recorded baseline")
        baseline = score(baseline_norm, head_rows)
        require(baseline["bf16_scores"] == identities[name]["observed_pair"], "baseline scores differ")
        def replace(selection):
            row = frozen_replacement(updates, observed_states, other_updates, selection)
            return score(normalized(row, gamma, epsilon), head_rows)
        all_updates = set(range(len(updates)))
        require(frozen_replacement(updates, observed_states, other_updates, all_updates) == other_states[-1],
                "using all saved peer updates did not reproduce the peer residual")
        layer_results = []
        for layer in range(layers):
            attention, mlp = 2 * layer, 2 * layer + 1
            layer_results.append({"layer": layer, "attention": replace({attention}), "mlp": replace({mlp}),
                                  "both": replace({attention, mlp})})
        selected = {2 * layer + part for layer in TARGET_LAYERS for part in (0, 1)}
        output[name] = {
            "capture_sha256": identities[name]["captured_rows_sha256"],
            "observed_selected_token_id": identities[name]["selected_id"],
            "baseline": baseline,
            "layers": layer_results,
            "control_replacements": {
                "all_attention": replace(set(range(0, 2 * layers, 2))),
                "all_mlp": replace(set(range(1, 2 * layers, 2))),
                "all_attention_and_mlp": replace(all_updates),
                "layers_21_22_27_attention_and_mlp": replace(selected),
            },
        }
    return output


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("previous_replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    shape_receipt = json.loads(args.config.read_bytes())
    previous = json.loads(args.previous_replay.read_bytes())
    shape = tuple(shape_receipt[key] for key in ("hidden_size", "vocab_size", "num_hidden_layers", "rms_norm_eps"))
    require(shape[:3] == (3584, 152064, 28) and not shape_receipt["tie_word_embeddings"]
            and math.isfinite(shape[3]) and 0 < shape[3] < 1, "unexpected checkpoint shape")
    head, gamma = bind_weights(args.weights, shape[0], previous)
    fixture = gzip.decompress(args.fixture.read_bytes()) if args.fixture.suffix == ".gz" else args.fixture.read_bytes()
    report = {
        "interpretation": "Uses captured updates as fixed values, replaying only the residual additions and final norm.",
        "probe_token_ids": TOKEN_IDS,
        "checkpoint_manifest_sha256": previous["checkpoint_manifest_sha256"],
        "config": shape_receipt,
        "previous_replay_sha256": digest(args.previous_replay.read_bytes()),
        "fixture_sha256": digest(fixture),
        "cases": {str(case): replay_case(args.trace, fixture, case, shape, previous, head, gamma) for case in (3, 9)},
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
