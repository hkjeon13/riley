"""Compare fused and separate projections of the same captured BF16 operands."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import struct

from trace_decode_layers import HIDDEN, difference, read_payload, require

CASES = ("669-9", "1023-67")
PARTS = (("q", 3584), ("k", 512), ("v", 512))
OUT = sum(width for _, width in PARTS)
REFERENCES = {
    "operators": "b7c459a5558d1d371228c751b83ad6bf29aeed4ff6100f3d472300bf59ece10a",
    "riley_plan": "77996ecda4a6b148504f353867fded9afc63cfca861a99e3818aaccc7b70e791",
    "hf_profile": "1a2e5f014dfc13b6cc28f3aa5da16f0ed730f6a61173e90d6bc2c805756caeed",
    "checkpoint": "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f",
}


def digest(data):
    return sha256(data).hexdigest()


def read_proof(file, role):
    raw = file.read_bytes()
    require(digest(raw) == REFERENCES[role], f"{role} reference digest differs")
    return json.loads(raw)


def write(path, payload):
    with path.open("xb") as file:
        file.write(payload)


def projection_weight(model, names, part, rows):
    name = f"model.layers.0.self_attn.{part}_proj.weight"
    filename = names[name]
    require(Path(filename).name == filename and filename.endswith(".safetensors"), "unexpected weight shard")
    file = model / filename
    with file.open("rb") as stream:
        extent, = struct.unpack("<Q", stream.read(8))
        require(0 < extent <= 16 * 1024**2, "invalid tensor header extent")
        header = stream.read(extent)
        require(len(header) == extent, "truncated tensor header")
        metadata = json.loads(header)[name]
        start, end = metadata["data_offsets"]
        require(metadata["dtype"] == "BF16" and metadata["shape"] == [rows, HIDDEN]
                and 0 <= start < end <= file.stat().st_size - 8 - extent
                and end - start == rows * HIDDEN * 2, "unexpected QKV operand shape")
        stream.seek(8 + extent + start)
        weight = stream.read(end - start)
    require(len(weight) == rows * HIDDEN * 2, "truncated weight")
    return weight, {"shard": filename, "tensor_sha256": digest(weight), "header_sha256": digest(header)}


def prepare(operands, model, operators, riley_plan, hf_profile):
    read_proof(model / "riley-checkpoint.json", "checkpoint")
    captured = read_proof(operators / "verified.json", "operators")
    selected = read_proof(riley_plan / "verified.json", "riley_plan")
    profiled = read_proof(hf_profile / "verified.json", "hf_profile")
    require(set(captured["cases"]) == set(selected["cases"]) == set(profiled["cases"]) == set(CASES),
            "different decode cases in frozen references")
    operands.mkdir()
    output = {}
    for name in CASES:
        case = captured["cases"][name]
        raw = {"riley": operators / "riley/rows" / name / "rows.bf16", "hf": operators / "hf/rows" / f"{name}.bf16"}
        raw = {arm: path.read_bytes() for arm, path in raw.items()}
        require(case["source_trace_sha256"] == {arm: digest(b) for arm, b in raw.items()},
                "operator trace files differ")
        require(case["source_trace_sha256"]["riley"] == selected["cases"][name]["identical_operator_trace_sha256"]
                and case["source_trace_sha256"]["hf"] == profiled["cases"][name]["identical_hf_trace_sha256"],
                "selected plans are bound to a different decoded token")
        expected = (1, OUT, HIDDEN, 21, 15, 18, 1, 0, 0, 0, 0, 12040, 120405, "8.0")
        fields = ("m", "n", "k", "id", "tile", "stages", "split_k", "reduction", "cta_swizzling", "custom_option",
                  "workspace", "cuda", "cublaslt", "sm")
        require(tuple(selected["cases"][name]["gemm"][key] for key in fields) == expected,
                "recorded Riley plan differs from isolated replay")
        decoded = {arm: read_payload(values) for arm, values in raw.items()}
        norm = decoded["riley"]["input_norm", "0"]
        require(norm == decoded["hf"]["input_norm", "0"], "the two captured operands differ")
        payloads = {"input": norm, **{arm: rows["qkv_prebias", "0"] for arm, rows in decoded.items()}}
        for kind, payload in payloads.items():
            write(operands / f"{name}.{kind}.bf16", payload)
        output[name] = {kind: digest(payload) for kind, payload in payloads.items()}
    index = (model / "model.safetensors.index.json").read_bytes()
    names = json.loads(index)["weight_map"]
    weights = {key: projection_weight(model, names, key, size) for key, size in PARTS}
    packed = b"".join(weights[key][0] for key, _ in PARTS)
    write(operands / "weights.bf16", packed)
    return {"references": REFERENCES, "cases": output, "packed_weights_sha256": digest(packed),
            "weights_index_sha256": digest(index), "projection_weights": {k: pair[1] for k, pair in weights.items()}}


def loaded(file, checksum, width):
    raw = file.read_bytes()
    require(len(raw) == 2 * width and digest(raw) == checksum, f"{file.name} identity changed")
    return raw


def profile_merged(norm, weight):
    from torch.nn.functional import linear
    from torch.profiler import ProfilerActivity, profile, record_function
    from profile_hf_qkv import children

    label = "W4_HF_MERGED"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as profiler:
        with record_function(label):
            result = linear(norm, weight)
    matches = [item for item in profiler.events() if item.name == label and str(item.device_type) == "DeviceType.CPU"]
    require(len(matches) == 1, "expected one merged host profile region")
    kernels = [kernel.name for event in children(matches[0]) if event.name == "aten::mm" for kernel in event.kernels]
    require(kernels, "no CUDA kernel associated with merged aten::mm")
    return result, kernels


def torch_replay(operands, preparation):
    import os
    import torch
    from profile_hf_qkv import capture

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0"
            and torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (8, 0),
            "expected only the reserved A100 GPU 7")
    prep = json.loads(preparation.read_bytes())
    raw_weights = loaded(operands / "weights.bf16", prep["packed_weights_sha256"], HIDDEN * OUT)
    tensor = torch.frombuffer(bytearray(raw_weights), dtype=torch.bfloat16).reshape(OUT, HIDDEN)
    merged = tensor.cuda()
    separate = [part.contiguous().clone() for part in merged.split([w for _, w in PARTS], dim=0)]
    slices = list(merged.split([w for _, w in PARTS], dim=0))
    results = {}
    with torch.inference_mode():
        for name in CASES:
            raw = loaded(operands / f"{name}.input.bf16", prep["cases"][name]["input"], HIDDEN)
            norm = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(1, 1, HIDDEN).cuda()
            split, split_profile = capture(norm, separate)
            sliced, sliced_profile = capture(norm, slices)
            joined, merged_kernels = profile_merged(norm, merged)
            routes = {"torch-separate": torch.cat(split, dim=-1), "torch-sliced": torch.cat(sliced, dim=-1),
                      "torch-merged": joined}
            result_hashes = {}
            for route, tensor_out in routes.items():
                require(tensor_out.shape == (1, 1, OUT) and tensor_out.dtype == torch.bfloat16, "unexpected projection result")
                blob = tensor_out.contiguous().view(torch.int16).cpu().numpy().tobytes()
                write(operands / f"{name}.{route}.bf16", blob)
                result_hashes[route] = digest(blob)
            results[name] = {"sha256": result_hashes, "separate_profile": split_profile,
                             "sliced_profile": sliced_profile, "merged_kernels": merged_kernels}
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cases": results}


def compare(operands, preparation, torch_report):
    prep = json.loads(preparation.read_bytes())
    torch = json.loads(torch_report.read_bytes())
    require(prep["references"] == REFERENCES and set(prep["cases"]) == set(torch["cases"]) == set(CASES),
            "replay receipts differ")
    loaded(operands / "weights.bf16", prep["packed_weights_sha256"], OUT * HIDDEN)
    cases = {}
    for name in CASES:
        expected = prep["cases"][name]
        loaded(operands / f"{name}.input.bf16", expected["input"], HIDDEN)
        baseline = {arm: loaded(operands / f"{name}.{arm}.bf16", expected[arm], OUT) for arm in ("riley", "hf")}
        actual = {route: loaded(operands / f"{name}.{route}.bf16", sha, OUT)
                  for route, sha in torch["cases"][name]["sha256"].items()}
        require(set(actual) == {"torch-separate", "torch-sliced", "torch-merged"}, "missing torch projection route")
        native = (operands / f"{name}.native-lt.bf16").read_bytes()
        require(len(native) == 2 * OUT, "native replay has incorrect size")
        actual["native-lt"] = native
        comparison = {key: {"sha256": digest(raw), **{arm: difference(raw, ref) for arm, ref in baseline.items()}}
                      for key, raw in actual.items()}
        cases[name] = {"original_riley_vs_hf": difference(baseline["riley"], baseline["hf"]), "replays": comparison,
                       "torch_kernels": {key: torch["cases"][name][key]
                                         for key in ("separate_profile", "sliced_profile", "merged_kernels")}}
    return {"scope": "Exact operand crossover for the captured layer-zero bias-free projections.",
            "binding": prep, "torch": torch["torch"], "torch_cuda": torch["cuda"], "cases": cases}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action, args in (("prepare", ("operands", "model", "operators", "riley_plan", "hf_profile")),
                         ("torch", ("operands", "preparation")),
                         ("compare", ("operands", "preparation", "torch_report"))):
        sub = commands.add_parser(action)
        for name in args:
            sub.add_argument(name, type=Path)
    options = vars(parser.parse_args())
    action = options.pop("action")
    result = {"prepare": prepare, "torch": torch_replay, "compare": compare}[action](**options)
    print(json.dumps(result, indent=2, allow_nan=False))
