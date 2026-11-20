"""Compare the recorded CUDA plans and candidate arithmetic against the same oracle."""
from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess

import torch
from safetensors import safe_open

from projection_oracle import (
    CASES, CHECKPOINT_SHA256, GPU, OPERATORS, checkpoint_shard, compare, digest,
    file_hash, numeric_words, operands, ref_dot, tensor_bytes,
)


FIELDS = ("id", "tile", "stages", "split_k", "reduction", "cta_swizzling", "custom_option", "flags", "workspace")


def read(path):
    return json.loads(path.read_bytes())


def pack(stage, model, source):
    shard, checksum = checkpoint_shard(model)
    records = operands(source)
    weight_hashes = {}
    for operator, (key, _, _, _, width) in OPERATORS.items():
        directory = stage / operator
        directory.mkdir()
        with safe_open(shard, framework="pt", device="cpu") as handle:
            weights = handle.get_tensor(key)
            assert weights.shape == (3584, width)
            raw = tensor_bytes(weights)
        weight_hashes[operator] = digest(raw)
        (directory / "weights.bf16").write_bytes(raw)
        for case in CASES:
            inputs, riley, _ = records[case][operator]
            (directory / f"{case}.input.bf16").write_bytes(inputs)
            (directory / f"{case}.riley.bf16").write_bytes(riley)
    print(json.dumps({"checkpoint_manifest_sha256": CHECKPOINT_SHA256,
                      "shard": shard.name, "shard_sha256": checksum,
                      "weight_tensor_sha256": weight_hashes}, indent=2))


def run_gpu(stage, native):
    plans = read(native / "plans.json")["cases"]
    for operator in OPERATORS:
        plan = plans[CASES[0]][operator]
        args = [plan["n"], plan["k"], plan["workspace"], *(plan[key] for key in FIELDS[:-1])]
        with (stage / operator / "plans.log").open("xb") as log:
            subprocess.run([str(stage / "replay_projection_plans"), str(stage / operator), *map(str, args)],
                           check=True, stdout=log, timeout=180)


def evaluate(stage, native, oracle, model, source):
    torch.set_num_threads(4)
    checkpoint = read(stage / "weights.json")
    precision = read(oracle / "verified.json")
    assert checkpoint["shard_sha256"] == precision["shard_sha256"]
    records = operands(source)
    captured = read(native / "plans.json")
    result = {}
    with safe_open(model / checkpoint["shard"], framework="pt", device="cpu") as handle:
        for operator, (key, _, _, _, _) in OPERATORS.items():
            directory = stage / operator
            weights = handle.get_tensor(key)
            assert digest(tensor_bytes(weights)) == checkpoint["weight_tensor_sha256"][operator]
            intro, *logs = (directory / "plans.log").read_text().splitlines()
            assert intro.startswith("heuristics ")
            metadata = {}
            for line in logs:
                label, *config = line.split()
                assert label not in metadata and len(config) == len(FIELDS)
                metadata[label] = dict(zip(FIELDS, map(int, config), strict=True))
            recorded = captured["cases"][CASES[0]][operator]
            assert metadata["recorded"] == {field: recorded[field] for field in FIELDS}
            results = {}
            for case in CASES:
                inputs, riley, _ = records[case][operator]
                assert (directory / f"{case}.input.bf16").read_bytes() == inputs
                assert (directory / f"{case}.recorded.bf16").read_bytes() == riley
                cpu, approx, fraction, _, _ = ref_dot(weights, numeric_words(inputs))
                base = precision["comparisons"][f"{case}/{operator}"]
                assert digest(cpu.astype("<u2", copy=False).tobytes()) == base["fp64_nearest_even_bf16_sha256"]
                assert digest(inputs) == base["input_sha256"]
                results[case] = {}
                for label in metadata:
                    raw = (directory / f"{case}.{label}.bf16").read_bytes()
                    results[case][label] = {
                        "sha256": digest(raw), **compare(numeric_words(raw), cpu, approx, fraction),
                    }
                assert results[case]["recorded"]["different_numeric_values"] == base["observations"]["riley"]["different_numeric_values"]
            result[operator] = {"heuristic_result_count": int(intro.split()[1]), "plans": metadata, "cases": results}
    print(json.dumps({
        "gpu_uuid": GPU, "checkpoint": checkpoint, "riley_capture": captured,
        "plan_capture_manifest_sha256": file_hash(native / "result.sha256"),
        "cpu_oracle_manifest_sha256": file_hash(oracle / "result.sha256"),
        "operators": result,
    }, indent=2, allow_nan=False))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pack", "gpu", "evaluate"))
    for name in ("stage", "native", "oracle", "model", "source"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.phase == "pack":
        pack(args.stage, args.model, args.source)
    elif args.phase == "gpu":
        run_gpu(args.stage, args.native)
    else:
        evaluate(args.stage, args.native, args.oracle, args.model, args.source)


if __name__ == "__main__":
    main()
