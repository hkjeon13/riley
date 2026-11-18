"""Recompute recorded BF16 residual additions and trace their output-margin change."""

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import struct

from replay_final_head import bf16, digest, read_arms, read_payload, read_weight_rows, require, round_bf16

STAGES = ("attention_projection", "attention_residual", "mlp_down", "layer_residual")
COMPONENTS = ("attention_projection", "attention_add_rounding", "mlp_down", "mlp_add_rounding")


def read_norm_weight(model, width):
    name = "model.norm.weight"
    index = json.loads((model / "model.safetensors.index.json").read_bytes())
    filename = index["weight_map"][name]
    require(Path(filename).name == filename and filename.endswith(".safetensors"), "unexpected norm shard")
    with (model / filename).open("rb") as stream:
        length, = struct.unpack("<Q", stream.read(8))
        require(0 < length <= 16 * 1024**2, "invalid norm shard header length")
        header = stream.read(length)
        entry = json.loads(header)[name]
        start, end = entry["data_offsets"]
        require(entry["dtype"] == "BF16" and entry["shape"] == [width]
                and 0 <= start < end <= (model / filename).stat().st_size - 8 - length
                and end - start == 2 * width, "unexpected final norm weight geometry")
        stream.seek(8 + length + start)
        values = stream.read(end - start)
    require(len(values) == 2 * width, "truncated final norm weight")
    return bf16(values), {"shard": filename, "header_sha256": digest(header), "weight_sha256": digest(values)}


def read_rows(trace, case, layers, width):
    native = trace / f"native{case}"
    hf = trace / f"hf{case}"
    receipts = [json.loads((directory / "receipt.json").read_bytes())["result_sha256"]
                for directory in (native, hf)]
    payloads = [read_payload(directory, "rows.bf16", receipt["rows.bf16"])
                for directory, receipt in zip((native, hf / "hf-rows"), receipts, strict=True)]
    _, *index = read_payload(native, "index.tsv", receipts[0]["index.tsv"]).decode().splitlines()
    rows = [{}, {}]
    expected_keys = {("embedding", "-")} | {(stage, str(layer)) for layer in range(layers) for stage in STAGES}
    offset = 0
    for line in index:
        stage, layer, start, length = line.split("\t")
        start, length = int(start), int(length)
        require(start == offset and length > 0 and length % 2 == 0, "invalid BF16 row extent")
        offset += length
        key = (stage, layer)
        if key in expected_keys:
            require(length == width * 2 and key not in rows[0], "duplicate or invalid residual row")
            for result, payload in zip(rows, payloads, strict=True):
                result[key] = bf16(payload[start:offset])
    require(rows[0].keys() == expected_keys and all(len(payload) == offset for payload in payloads),
            "incomplete residual trace")
    return rows


def dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def residual_path(rows, layers):
    previous = rows["embedding", "-"]
    for layer in range(layers):
        attention, after_attention, mlp, after_mlp = (rows[stage, str(layer)] for stage in STAGES)
        for base, update, observed in ((previous, attention, after_attention), (after_attention, mlp, after_mlp)):
            require(all(round_bf16(p + d) == x for p, d, x in zip(base, update, observed, strict=True)),
                    f"layer {layer}: captured residual addition differs from BF16 reimplementation")
        yield (
            attention,
            [x - (p + d) for p, d, x in zip(previous, attention, after_attention, strict=True)],
            mlp,
            [x - (p + d) for p, d, x in zip(after_attention, mlp, after_mlp, strict=True)],
        )
        previous = after_mlp


def decompose(rows, norms, raw_weights, gamma, epsilon, layers):
    before, after = rows
    head_difference = [b - a for a, b in zip(bf16(raw_weights[264]), bf16(raw_weights[5257]), strict=True)]
    direction = [g * d for g, d in zip(gamma, head_difference, strict=True)]
    final = [arm["layer_residual", str(layers - 1)] for arm in (before, after)]
    scale = [1 / math.sqrt(dot(row, row) / len(row) + epsilon) for row in final]
    average_scale = math.fsum(scale) / 2
    scores = [dot(row, direction) for row in final]
    embedding_delta = average_scale * (dot(after["embedding", "-"], direction)
                                       - dot(before["embedding", "-"], direction))
    by_layer = []
    components = zip(residual_path(before, layers), residual_path(after, layers), strict=True)
    for layer, (old, new) in enumerate(components):
        change = [(dot(b, direction) - dot(a, direction)) * average_scale for a, b in zip(old, new, strict=True)]
        by_layer.append({"layer": layer, **dict(zip(COMPONENTS, change, strict=True)), "total": math.fsum(change)})
    path_delta = math.fsum([embedding_delta, *(row["total"] for row in by_layer)])
    require(math.isclose(path_delta, average_scale * (scores[1] - scores[0]), abs_tol=1e-10),
            "residual path projections do not close")
    scale_delta = (scale[1] - scale[0]) * math.fsum(scores) / 2
    fp64_norm_change = scale[1] * scores[1] - scale[0] * scores[0]
    require(math.isclose(path_delta + scale_delta, fp64_norm_change, abs_tol=1e-10),
            "FP64 final-norm decomposition does not close")
    observed_change = dot(norms["hf"], head_difference) - dot(norms["riley"], head_difference)
    norm_checks = {}
    for engine, row, inverse_rms in zip(("riley", "hf"), final, scale, strict=True):
        staged = [round_bf16(round_bf16(x * inverse_rms) * g) for x, g in zip(row, gamma, strict=True)]
        single = [round_bf16(x * inverse_rms * g) for x, g in zip(row, gamma, strict=True)]
        observed = norms[engine]
        norm_checks[engine] = {
            "staged_matches": sum(x == y for x, y in zip(staged, observed, strict=True)),
            "single_round_matches": sum(x == y for x, y in zip(single, observed, strict=True)),
            "elements": len(observed),
            "staged_margin_error": dot(staged, head_difference) - dot(observed, head_difference),
        }
    return {"verified_bf16_additions_per_engine": 2 * layers * len(direction),
            "observed_norm_margin_delta_hf_minus_riley": observed_change,
            "residual_path_projection_delta": path_delta,
            "fp64_norm_scale_delta": scale_delta,
            "observed_norm_remainder": observed_change - fp64_norm_change,
            "fp64_reduction_final_norm_rounding_controls": norm_checks,
            "embedding_delta": embedding_delta,
            "component_totals": {key: math.fsum(row[key] for row in by_layer) for key in COMPONENTS},
            "layers": by_layer}


def replay(model, trace):
    config = json.loads((model / "config.json").read_bytes())
    width, vocab, layers = (config[key] for key in ("hidden_size", "vocab_size", "num_hidden_layers"))
    require((width, vocab, layers) == (3584, 152064, 28) and config["tie_word_embeddings"] is False,
            "expected the traced Qwen2.5-7B geometry")
    epsilon = config["rms_norm_eps"]
    require(0 < epsilon < 1 and math.isfinite(epsilon), "invalid RMSNorm epsilon")
    checkpoint_sha = digest((model / "riley-checkpoint.json").read_bytes())
    raw, weights = read_weight_rows(model, width, vocab)
    gamma, norm_weight = read_norm_weight(model, width)
    fixture = (trace / "fixtures/legacy.json").read_bytes()
    cases = {}
    for case in (3, 9):
        norms, identity = read_arms(trace, fixture, case, width, vocab, checkpoint_sha)
        result = decompose(read_rows(trace, case, layers, width), norms, raw, gamma, epsilon, layers)
        cases[str(case)] = {**result, "capture_sha256": {
            engine: record["captured_rows_sha256"] for engine, record in identity.items()}}
    return {"checkpoint_manifest_sha256": checkpoint_sha, "weights": weights, "norm_weight": norm_weight,
            "interpretation": "Projection of recorded residual increments onto the final token-pair margin direction",
            "cases": cases}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.model, args.trace), indent=2, allow_nan=False))
