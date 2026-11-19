"""Recompute only the mismatched layer-zero QKV products with FP64 summation."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import struct

from replay_final_head import bf16, round_bf16
from trace_decode_layers import HIDDEN, digest, read_payload, require


PARTS = (("q", 3584), ("k", 512), ("v", 512))
TENSOR = "model.layers.0.self_attn.{}_proj.weight"


def weight_row(model, index, part, width, offset):
    name = TENSOR.format(part)
    filename = index[name]
    require(Path(filename).name == filename and filename.endswith(".safetensors"), "unexpected tensor shard")
    file = model / filename
    with file.open("rb") as handle:
        raw_length = handle.read(8)
        require(len(raw_length) == 8, "missing tensor header length")
        length, = struct.unpack("<Q", raw_length)
        require(0 < length <= 16 * 1024**2, "invalid tensor header length")
        header = handle.read(length)
        entry = json.loads(header)[name]
        start, end = entry["data_offsets"]
        require(entry["dtype"] == "BF16" and entry["shape"] == [width, HIDDEN]
                and 0 <= start < end <= file.stat().st_size - length - 8 and end - start == 2 * width * HIDDEN
                and 0 <= offset < width, "QKV weight row geometry differs")
        handle.seek(length + 8 + start + 2 * offset * HIDDEN)
        raw = handle.read(2 * HIDDEN)
    require(len(raw) == 2 * HIDDEN, "truncated QKV weight row")
    return raw, {"shard": filename, "header_sha256": digest(header), "row_sha256": digest(raw)}


def compare(model, stage):
    raw_result = (stage / "verified.json").read_bytes()
    result = json.loads(raw_result)
    index_bytes = (model / "model.safetensors.index.json").read_bytes()
    weights = json.loads(index_bytes)["weight_map"]
    cases = {}
    for name, expected in result["cases"].items():
        native_raw = (stage / "riley/rows" / name / "rows.bf16").read_bytes()
        hf_raw = (stage / "hf/rows" / f"{name}.bf16").read_bytes()
        require(expected["preexisting_endpoints_exact"] and expected["source_trace_sha256"] ==
                {"riley": digest(native_raw), "hf": digest(hf_raw)}, "captured trace identity changed")
        riley, hf = map(read_payload, (native_raw, hf_raw))
        norm, key = ("input_norm", "0"), ("qkv_prebias", "0")
        require(riley[norm] == hf[norm], "projection input differs")
        vec = bf16(riley[norm])
        left, right = bf16(riley[key]), bf16(hf[key])
        output = {}
        cursor = 0
        for part, width in PARTS:
            observed = []
            for row in range(width):
                pos = cursor + row
                if left[pos] == right[pos]:
                    continue
                raw, proof = weight_row(model, weights, part, width, row)
                exact = math.fsum(a * b for a, b in zip(vec, bf16(raw), strict=True))
                before, after = left[pos], right[pos]
                midpoint = (before + after) / 2
                fp32_ulp = math.ldexp(1.0, max(math.frexp(exact)[1] - 24, -149))
                rounded = round_bf16(exact)
                observed.append({
                    "row": row, "riley_bf16": before, "hf_bf16": after,
                    "fp64_dot": exact, "fp64_rounded_bf16": rounded,
                    "adjacent_bf16_values": abs(int.from_bytes(riley[key][pos*2:pos*2+2], "little")
                                                - int.from_bytes(hf[key][pos*2:pos*2+2], "little")) == 1,
                    "distance_to_observed_midpoint_in_fp32_ulps": abs(exact - midpoint) / fp32_ulp,
                    "rounding_matches": [label for label, value in (("riley", before), ("hf", after)) if value == rounded],
                    "weight": proof,
                })
            output[part] = observed
            cursor += width
        count = sum(len(v) for v in output.values())
        require(count == expected["earliest_observed_difference"]["different_bf16_words"],
                "mismatched products do not reproduce the operator comparison")
        cases[name] = {"same_norm_sha256": digest(riley[norm]), "mismatched_products": count, "projections": output}
    return {
        "scope": "FP64 dot products for the observed BF16 discrepancies at bias-free layer-zero QKV.",
        "trace_summary_sha256": sha256(raw_result).hexdigest(),
        "checkpoint_manifest_sha256": digest((model / "riley-checkpoint.json").read_bytes()),
        "weights_index_sha256": digest(index_bytes),
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.model,args.stage),indent=2,allow_nan=False))
