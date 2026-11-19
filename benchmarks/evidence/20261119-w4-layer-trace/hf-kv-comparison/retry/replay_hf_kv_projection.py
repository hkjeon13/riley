"""Replay the final input row's layer-0 K/V bias placement from captured inputs."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open
import torch

from replay_attention_mask import capture


HIDDEN = 3584
HEAD = 128
KV = 512


def bits(tensor):
    return tensor.contiguous().view(torch.uint16).numpy().tobytes()


def bf16(data):
    return torch.from_numpy(np.frombuffer(data, dtype="<u2").copy()).view(torch.bfloat16)


def compare(expected, actual):
    delta = expected.float() - actual.float()
    equal = expected.view(torch.uint16) == actual.view(torch.uint16)
    return {"elements": int(equal.numel()), "different_bf16_elements": int((~equal).sum()),
            "max_abs": float(delta.abs().max())}


def rotate(key, theta, position):
    exponents = torch.arange(0, HEAD, 2, dtype=torch.float32) / HEAD
    angles = position * (1.0 / (theta ** exponents))
    cosine = angles.cos().repeat(2).bfloat16()
    sine = angles.sin().repeat(2).bfloat16()
    left, right = key.chunk(2)
    return key * cosine + torch.cat((-right, left)) * sine


def load_projection(model, weight_map, projection):
    prefix = f"model.layers.0.self_attn.{projection}_proj"
    pieces = {}
    for label, shape in (("weight", (KV, HIDDEN)), ("bias", (KV,))):
        name = f"{prefix}.{label}"
        file = weight_map[name]
        if Path(file).name != file:
            raise ValueError("unexpected checkpoint tensor path")
        with safe_open(model / file, framework="pt", device="cpu") as shard:
            view = shard.get_slice(name)
            if tuple(view.get_shape()) != shape:
                raise ValueError(f"unexpected shape for {name}")
            tensor = view[:HEAD]
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"unexpected dtype for {name}")
        pieces[label] = tensor
    return pieces["weight"], pieces["bias"]


def replay(model, stage, native, reference):
    checkpoint = sha256((model / "riley-checkpoint.json").read_bytes()).hexdigest()
    verified = json.loads((stage / "result.json").read_bytes())
    config = json.loads((model / "config.json").read_bytes())
    if (checkpoint != verified["checkpoint_sha256"] or config["hidden_size"] != HIDDEN
        or config["num_key_value_heads"] != KV // HEAD or config.get("rope_scaling")):
        raise ValueError("unexpected checkpoint or model geometry")
    index = (model / "model.safetensors.index.json").read_bytes()
    weight_map = json.loads(index)["weight_map"]
    projections = {name: load_projection(model, weight_map, name) for name in ("k", "v")}
    cases = {}
    for case in verified["cases"]:
        number = case["case"]
        last = case["positions"][-1]
        rows, _, _, _, native_key, native_value, digest = capture(native / f"native{number}", last)
        if digest != case["riley_capture_sha256"]:
            raise ValueError("Riley KV capture changed")
        historic = (reference / f"hf{number}/hf-rows/rows.bf16").read_bytes()
        norm = rows[("input_norm", "0")]
        prebias = rows[("qkv_prebias", "0")]
        if historic[2 * HIDDEN:4 * HIDDEN] != norm or historic[4 * HIDDEN:4 * HIDDEN + len(prebias)] != prebias:
            raise ValueError("the engines differ before the tested bias operation")
        vectors = {}
        for projection, role, start, native_vector in (
            ("k", "key", HIDDEN, native_key[-1]), ("v", "value", HIDDEN + KV, native_value[-1]),
        ):
            raw = (stage / f"hf-case{number}/{role}.bf16").read_bytes()
            if sha256(raw).hexdigest() != case["hf_capture_sha256"][role]:
                raise ValueError("actual HF KV data changed")
            observed = {"riley": torch.from_numpy(native_vector).bfloat16(), "hf": bf16(raw[-2 * HEAD:])}
            stored = bf16(prebias)[start:start + HEAD]
            weight, bias = projections[projection]
            accumulated = weight.double() @ bf16(norm).double()
            orders = {"rounded_projection_then_bias": stored + bias,
                      "bias_then_rounded_projection": (accumulated + bias.double()).bfloat16()}
            if role == "key":
                orders = {name: rotate(data, config["rope_theta"], last) for name, data in orders.items()}
            vectors[role] = {"before_bias_fp64_vs_captured_bf16": compare(accumulated.bfloat16(), stored),
                             "replay_vs_engine": {order: {engine: compare(prediction, capture)
                                                         for engine, capture in observed.items()}
                                                  for order, prediction in orders.items()}}
        cases[str(number)] = {"position": last, "native_capture_sha256": digest, "vectors": vectors}
    return {"checkpoint_sha256": checkpoint, "index_sha256": sha256(index).hexdigest(),
            "weight_rows_sha256": {name: {"weight": sha256(bits(weight)).hexdigest(),
                                         "bias": sha256(bits(bias)).hexdigest()}
                                    for name, (weight, bias) in projections.items()},
            "torch": str(torch.__version__), "layer": 0, "kv_head": 0, "cases": cases}


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(replay(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
