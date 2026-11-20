"""Bind observed projections to stored BF16 operands and independently round their dot products."""
from argparse import ArgumentParser
from bisect import bisect_left
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn.functional import linear


CHECKPOINT_SHA256 = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
MATCHED_RESULT_SHA256 = "3df934c46b0a8fb526c2d9d926ad5e5267b6334cd8dc6e2f62985eab9bb2635f"
GPU = "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0"
OPERATORS = {
    "o_proj": ("model.layers.0.self_attn.o_proj.weight", "attention_context", "attention_projection",
               "attention_context", 3584),
    "down_proj": ("model.layers.0.mlp.down_proj.weight", "mlp_gated", "mlp_down", "same_gated_mlp_input", 18944),
}
CASES = ("669-9", "1023-67")
GROUPS = ("hf_default", "hf_no_reduced_bf16", "fp32_no_tf32")


def digest(data):
    return sha256(data).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def checkpoint_shard(model):
    manifest_path = model / "riley-checkpoint.json"
    assert file_hash(manifest_path) == CHECKPOINT_SHA256
    records = json.loads(manifest_path.read_bytes())["files"]
    index = json.loads((model / "model.safetensors.index.json").read_bytes())["weight_map"]
    filenames = {index[name] for name, *_ in OPERATORS.values()}
    assert len(filenames) == 1
    name, = filenames
    entry, = (item for item in records if item["path"] == name)
    shard = model / name
    assert shard.stat().st_size == entry["bytes"] and file_hash(shard) == entry["sha256"]
    return shard, entry["sha256"]


def read_rows(path):
    raw = (path / "rows.bf16").read_bytes()
    _, *index = (path / "index.tsv").read_text().splitlines()
    result = {}
    offset = 0
    for line in index:
        name, layer, start, width = line.split("\t")
        assert int(start) == offset and int(width) > 0 and (name, layer) not in result
        end = offset + int(width)
        result[name, layer] = raw[offset:end]
        offset = end
    assert offset == len(raw)
    return raw, result


def operands(source):
    reference = (source / "hf/verified.json").read_bytes()
    assert digest(reference) == MATCHED_RESULT_SHA256
    cases = json.loads(reference)["cases"]
    out = {}
    for name in CASES:
        case = cases[name]
        row, values = read_rows(source / "riley/rows" / name)
        assert digest(row) == case["native_source_sha256"]["new_operator_trace"]
        out[name] = {}
        for operator, (_, input_point, output_point, trial_name, width) in OPERATORS.items():
            trial = case["trials"][trial_name]["operators_vs_riley"]
            input_bytes, output_bytes = values[input_point, "0"], values[output_point, "0"]
            assert len(input_bytes) == width * 2 and len(output_bytes) == 3584 * 2
            assert digest(input_bytes) == trial[input_point]["effective_sha256"] == trial[input_point]["riley_sha256"]
            assert digest(output_bytes) == trial[output_point]["riley_sha256"]
            out[name][operator] = (input_bytes, output_bytes, trial[output_point]["emitted_sha256"])
    return out


def tensor_bytes(value):
    assert value.dtype == torch.bfloat16
    return value.cpu().contiguous().view(torch.uint16).numpy().tobytes()


def evaluate_gpu(stage, shard, records, shard_hash):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == GPU and torch.cuda.device_count() == 1
    assert torch.__version__ == "2.13.0+cu129"
    stage.joinpath("gpu").mkdir()
    torch.set_num_threads(4)
    matmul = torch.backends.cuda.matmul
    original_reduction = matmul.allow_bf16_reduced_precision_reduction
    original_precision = torch.get_float32_matmul_precision()
    defaults = str(original_reduction)
    result = {}
    with safe_open(shard, framework="pt", device="cpu") as handle, torch.inference_mode():
        for operator, (key, _, _, _, width) in OPERATORS.items():
            weights = handle.get_tensor(key)
            assert weights.shape == (3584, width) and weights.dtype == torch.bfloat16
            gpu_weights = weights.cuda()
            for case in CASES:
                input_bytes, _, expected = records[case][operator]
                vector = torch.frombuffer(bytearray(input_bytes), dtype=torch.bfloat16).reshape(1, 1, width).cuda()
                assert tensor_bytes(vector) == input_bytes
                try:
                    default = tensor_bytes(linear(vector, gpu_weights))
                    assert digest(default) == expected, f"{case} {operator}: standalone HF does not reproduce the serving trace"
                    matmul.allow_bf16_reduced_precision_reduction = False
                    full = tensor_bytes(linear(vector, gpu_weights))
                    torch.set_float32_matmul_precision("highest")
                    assert not matmul.allow_tf32
                    converted = tensor_bytes(linear(vector.float(), gpu_weights.float()).bfloat16())
                finally:
                    matmul.allow_bf16_reduced_precision_reduction = original_reduction
                    torch.set_float32_matmul_precision(original_precision)
                assert tensor_bytes(linear(vector, gpu_weights)) == default
                hashes = {}
                for group, raw in zip(GROUPS, (default, full, converted), strict=True):
                    with (stage / "gpu" / f"{case}-{operator}-{group}.bf16").open("xb") as out:
                        out.write(raw)
                    hashes[group] = digest(raw)
                result[f"{case}/{operator}"] = hashes
    print(json.dumps({"shard_sha256": shard_hash, "torch": torch.__version__, "gpu_uuid": GPU,
                      "default_bf16_reduced_precision_allowed": defaults, "device_outputs": result},
                     indent=2, allow_nan=False))


def units(bits):
    exponent = (bits >> 7) & 255
    sign = -1 if bits & 0x8000 else 1
    assert exponent < 255
    mantissa = bits & 127
    return sign * ((128 + mantissa) << (exponent - 1) if exponent else mantissa)


def rounding_grid():
    finite = np.arange(0x7f80, dtype=np.uint32)
    values = (finite << 16).view(np.float32).astype(np.float64)
    midpoints = (values[:-1] + values[1:]) / 2
    bf16_units = [units(int(code)) for code in finite]
    scaled_midpoints = [(a + b) << 132 for a, b in zip(bf16_units, bf16_units[1:])]
    return midpoints, scaled_midpoints


def exact_word(weight_bits, vector_units, scaled_midpoints):
    dot = sum(units(int(weight)) * value for weight, value in zip(weight_bits, vector_units, strict=True))
    magnitude = abs(dot)
    slot = bisect_left(scaled_midpoints, magnitude)
    tie = slot < len(scaled_midpoints) and magnitude == scaled_midpoints[slot]
    if tie and slot & 1:
        slot += 1
    return slot | (0x8000 if dot < 0 else 0), tie


def absolute_midpoint_fraction(magnitude, rounded, midpoints):
    lo = midpoints[np.maximum(rounded - 1, 0)]
    hi = midpoints[np.minimum(rounded, len(midpoints) - 1)]
    return np.minimum(np.abs(magnitude - lo), np.abs(magnitude - hi)) / np.maximum(hi - lo, midpoints[0])


def numeric_words(data):
    return np.frombuffer(data, dtype="<u2")


def as_float(words):
    return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def ref_dot(weights, vector_bits):
    midpoints, scaled = rounding_grid()
    input64 = as_float(vector_bits)
    input_units = [units(int(bits)) for bits in vector_bits]
    reference = []
    exact_values = []
    fractions = []
    fallbacks = ties = 0
    for slab in weights.split(128):
        slab64 = slab.double().numpy()
        value = slab64 @ input64
        magnitude = np.abs(value)
        assert np.isfinite(value).all() and (magnitude <= midpoints[-1]).all()
        rounded = np.searchsorted(midpoints, magnitude, side="left")
        fraction = absolute_midpoint_fraction(magnitude, rounded, midpoints)
        lo = midpoints[np.maximum(rounded - 1, 0)]
        hi = midpoints[rounded]
        distance = np.minimum(np.abs(magnitude - lo), np.abs(magnitude - hi))
        # Bound the FP64 dot-product error using the absolute-product sum. The
        # factor of four also covers rounding while calculating that bound.
        bound = (4 * (weights.shape[1] + 2) * np.finfo(np.float64).eps
                 * (np.abs(slab64) @ np.abs(input64)))
        encoded = (rounded | np.where(value < 0, 0x8000, 0)).astype(np.uint16)
        for index in np.flatnonzero(distance <= bound):
            word, tie = exact_word(slab[index].view(torch.uint16).numpy(), input_units, scaled)
            encoded[index] = word
            fallbacks += 1
            ties += tie
        reference.append(encoded)
        exact_values.append(value)
        fractions.append(fraction)
    return np.concatenate(reference), np.concatenate(exact_values), np.concatenate(fractions), fallbacks, ties


def ordered(words):
    magnitude = words.astype(np.int32) & 0x7fff
    return np.where(words & 0x8000, -magnitude, magnitude)


def compare(observed, reference, approx, fractions):
    assert observed.shape == reference.shape and np.isfinite(as_float(observed)).all()
    delta = np.abs(ordered(observed) - ordered(reference))
    bad = delta != 0
    error = as_float(observed) - approx
    return {
        "different_words_including_zero_sign": int(np.count_nonzero(observed != reference)),
        "different_numeric_values": int(np.count_nonzero(bad)),
        "more_than_one_bf16_step": int(np.count_nonzero(delta > 1)),
        "max_bf16_steps": int(delta.max()),
        "rmse_to_fp64": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs_error_to_fp64": float(np.max(np.abs(error))),
        "mismatches_within_0p001_local_bf16_step_of_midpoint": int(np.count_nonzero(bad & (fractions <= 0.001))),
        "max_mismatch_midpoint_distance_in_local_bf16_steps": float(fractions[bad].max()) if bad.any() else None,
    }


def evaluate_cpu(stage, shard, records, shard_hash):
    torch.set_num_threads(4)
    receipt = json.loads((stage / "gpu-receipt.json").read_bytes())
    assert receipt["shard_sha256"] == shard_hash and receipt["gpu_uuid"] == GPU
    report = {}
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for operator, (key, _, _, _, width) in OPERATORS.items():
            weights = handle.get_tensor(key)
            assert weights.shape == (3584, width) and weights.dtype == torch.bfloat16
            for case in CASES:
                input_bytes, output_bytes, expected = records[case][operator]
                label = f"{case}/{operator}"
                observed = {"riley": output_bytes}
                for group in GROUPS:
                    raw = (stage / "gpu" / f"{case}-{operator}-{group}.bf16").read_bytes()
                    assert digest(raw) == receipt["device_outputs"][label][group]
                    observed[group] = raw
                assert digest(observed["hf_default"]) == expected
                nearest, approx, fraction, fallbacks, ties = ref_dot(weights, numeric_words(input_bytes))
                report[label] = {
                    "shape_m_n_k": [1, 3584, width],
                    "input_sha256": digest(input_bytes),
                    "fp64_nearest_even_bf16_sha256": digest(nearest.astype("<u2", copy=False).tobytes()),
                    "exact_integer_tie_or_near_tie_fallback_rows": fallbacks,
                    "exact_midpoint_ties": ties,
                    "observations": {group: {"output_sha256": digest(data),
                                            **compare(numeric_words(data), nearest, approx, fraction)}
                                     for group, data in observed.items()},
                }
    print(json.dumps({"checkpoint_manifest_sha256": CHECKPOINT_SHA256,
                      "matched_operator_result_sha256": MATCHED_RESULT_SHA256,
                      "shard_sha256": shard_hash, "gpu_comparison": receipt, "comparisons": report},
                     indent=2, allow_nan=False))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("bind", "gpu", "cpu"))
    for key in ("stage", "source", "model"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    if args.phase == "bind":
        shard, checksum = checkpoint_shard(args.model)
        print(json.dumps({"filename": shard.name, "sha256": checksum}))
        return
    binding = json.loads((args.stage / "shard.json").read_bytes())
    records = operands(args.source)
    if args.phase == "gpu":
        evaluate_gpu(args.stage, args.model / binding["filename"], records, binding["sha256"])
    else:
        evaluate_cpu(args.stage, args.model / binding["filename"], records, binding["sha256"])


if __name__ == "__main__":
    main()
