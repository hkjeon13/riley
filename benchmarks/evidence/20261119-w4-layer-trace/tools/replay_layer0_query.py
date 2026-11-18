"""Replay the first divergent query from the pinned BF16 traces on the CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from layer_trace import read_native
from long_accuracy import save


def digest(tensor):
    return sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def bf16_row(data):
    return torch.from_numpy(np.frombuffer(data, dtype="<u2").copy()).view(torch.bfloat16)


def compare(actual, predicted):
    delta = (actual.float() - predicted.float()).abs()
    return {
        "different_bf16_elements": int((actual.view(torch.uint16) != predicted.view(torch.uint16)).sum()),
        "max_abs": float(delta.max()),
    }


def rotate(query, cosine, sine):
    heads = query.reshape(-1, cosine.numel())
    left, right = heads.chunk(2, dim=-1)
    return (heads * cosine + torch.cat((-right, left), dim=-1) * sine).flatten()


def replay(model_dir, fixture, case, native_dir, hf_dir):
    model = json.loads((model_dir / "config.json").read_bytes())
    source = json.loads(fixture.read_bytes())["baseline"][case]
    rows, header = read_native(native_dir, source["prompt_tokens"])
    hidden, heads = model["hidden_size"], model["num_attention_heads"]
    kv = model["num_key_value_heads"] * (hidden // heads)
    metadata = json.loads((hf_dir / "metadata.json").read_bytes())
    if (metadata["case"], metadata["prompt_sha256"]) != (case, source["prompt_sha256"]):
        raise ValueError("HF and Riley rows refer to different prompts")
    hf_rows = (hf_dir / "rows.bf16").read_bytes()
    # The trace begins with embedding, norm, merged QKV, and then rotary query.
    qkv_start = 2 * 2 * hidden
    rope_start = qkv_start + 2 * (hidden + 2 * kv)
    prebias = bf16_row(rows[("qkv_prebias", "0")])[:hidden]
    if not torch.equal(prebias, bf16_row(hf_rows[qkv_start:qkv_start + 2 * hidden])):
        raise ValueError("replay requires a common BF16 pre-bias query")
    observed = {
        "riley": bf16_row(rows[("query_rope", "0")]),
        "hf": bf16_row(hf_rows[rope_start:rope_start + 2 * hidden]),
    }
    tensors = json.loads((model_dir / "model.safetensors.index.json").read_bytes())["weight_map"]
    name = "model.layers.0.self_attn.q_proj"

    def get_tensor(key):
        with safe_open(model_dir / tensors[key], framework="pt", device="cpu") as shard:
            return shard.get_tensor(key)

    bias, weight = get_tensor(f"{name}.bias"), get_tensor(f"{name}.weight")
    if bias.shape != (hidden,) or weight.shape != (hidden, hidden) or bias.dtype != weight.dtype:
        raise ValueError("unexpected Q projection geometry")
    norm = bf16_row(rows[("input_norm", "0")])
    accumulated = weight.double() @ norm.double()
    queries = {
        "round_before_bias": prebias + bias,
        "bias_before_round": (accumulated + bias.double()).bfloat16(),
    }
    head_width = hidden // heads
    exponents = torch.arange(0, head_width, 2, dtype=torch.float32) / head_width
    angles = header["position"] * (1.0 / (model["rope_theta"] ** exponents))
    cosine = angles.cos().repeat(2).bfloat16()
    sine = angles.sin().repeat(2).bfloat16()
    return {
        "case": case,
        "position": header["position"],
        "head_width": head_width,
        "torch_cpu_version": torch.__version__,
        "weight_sha256": digest(weight),
        "bias_sha256": digest(bias),
        "cpu_accumulator_rounded_vs_trace_prebias": compare(prebias, accumulated.bfloat16()),
        "rotary_query_replay": {
            name: {engine: compare(data, rotate(query, cosine, sine)) for engine, data in observed.items()}
            for name, query in queries.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("case", type=int, choices=(3, 9))
    parser.add_argument("native_dir", type=Path)
    parser.add_argument("hf_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    save(args.output, replay(args.model, args.fixture, args.case, args.native_dir, args.hf_dir))


if __name__ == "__main__":
    main()
