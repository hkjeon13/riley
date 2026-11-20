"""Locate model arithmetic differences with matched layer-zero operands."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from capture_final_rows import CHECKPOINT, GPU, HIDDEN, VOCAB, checkpoint, row_bytes, stem, targets
import progression_trace
from replay_decode_boundaries import prefix_digest, scores
from replay_final_head import bf16, round_bf16
from trace_decode_layers import LAYERS, check_endpoint, difference, digest, position, prior_report, read_native


INTERMEDIATE = 18_944
BOUNDARY_TRACE_SHA256 = "396e59c9e9c34d56d34bf257679b49ee5a1108804a0f664fd056a93a86c89a06"
OPERATOR_TRACE_SHA256 = "b7c459a5558d1d371228c751b83ad6bf29aeed4ff6100f3d472300bf59ece10a"
POINTS = (
    "attention_context", "attention_projection", "attention_residual", "post_attention_norm",
    "mlp_gated", "mlp_down", "layer_residual",
)
MLP_POINTS = POINTS[3:]


def read_operator(directory):
    index = (directory / "index.tsv").read_text().splitlines()
    raw = (directory / "rows.bf16").read_bytes()
    cursor = 0
    rows = {}
    for line in index[1:]:
        stage, layer, start, size = line.split("\t")
        assert int(start) == cursor and (stage, layer) not in rows
        cursor += int(size)
        rows[stage, layer] = raw[int(start):cursor]
    assert cursor == len(raw)
    return rows


def verify_trace(stage, operator, progression, original, offset):
    name = stem(original, offset)
    fresh, riley = read_native(stage / "riley/rows" / name, original, offset)
    old = read_operator(operator / "riley/rows" / name)
    assert all(riley[key] == value for key, value in old.items()), "new trace changes an observed operator row"
    baseline_raw, baseline = progression_trace.read_native(
        progression / "riley/rows" / name, original, offset)
    assert riley["layer_residual", "0"] == baseline["layer_residual", "0"]
    receipt = json.loads((stage / "riley/client.log").read_bytes())
    assert receipt[name]["output_match"] and receipt[name]["trace_sha256"] == digest(fresh)
    return riley, {"new_operator_trace": digest(fresh), "prior_residual_trace": digest(baseline_raw)}


def summed_residual(base, delta, output):
    values = zip(bf16(base), bf16(delta), strict=True)
    return all(round_bf16(a + b) == c for (a, b), c in zip(values, bf16(output), strict=True))


def replay_case(model, stage, operator, progression, evidence, original, offset, other):
    name = stem(original, offset)
    riley, source_hashes = verify_trace(stage, operator, progression, original, offset)
    hf_trace = progression_trace.read_payload((progression / "hf/rows" / f"{name}.bf16").read_bytes())
    baseline = evidence["cases"][name]
    check_endpoint(riley, baseline["engines"]["riley"], "Riley")
    assert all(summed_residual(*operands) for operands in (
        (riley["embedding", "-"], riley["attention_projection", "0"], riley["attention_residual", "0"]),
        (riley["attention_residual", "0"], riley["mlp_down", "0"], riley["layer_residual", "0"]),
    ))
    riley_id = original["tokens"][offset]
    assert (riley_id, other) == (baseline["riley_token"], baseline["hf_token"])
    pos, token = position(original, offset)
    output = model(torch.tensor([original["prompt_tokens"]], device="cuda"),
                   use_cache=True, logits_to_keep=1)
    for prior in original["tokens"][:offset - 1]:
        output = model(torch.tensor([[prior]], device="cuda"),
                       past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    del output
    cache_hash = prefix_digest(cache, pos)
    current = torch.tensor([[token]], device="cuda")
    layer = model.model.layers[0]
    modules = {
        "attention_context": (layer.self_attn.o_proj, True),
        "attention_projection": (layer.self_attn.o_proj, False),
        "attention_residual": (layer.post_attention_layernorm, True),
        "post_attention_norm": (layer.post_attention_layernorm, False),
        "mlp_gated": (layer.mlp.down_proj, True),
        "mlp_down": (layer.mlp.down_proj, False),
        "layer_residual": (layer, False),
    }

    def trial(replace):
        effective = {}
        emitted = {}

        def tap(name, value):
            width = INTERMEDIATE if name == "mlp_gated" else HIDDEN
            assert name not in emitted
            raw = row_bytes(value, width)
            emitted[name] = raw
            if name in replace:
                value.copy_(torch.frombuffer(bytearray(replace[name]), dtype=torch.bfloat16)
                            .reshape(1, 1, width).cuda())
                raw = row_bytes(value, width)
                assert raw == replace[name]
            effective[name] = raw

        def hook(name, module, before):
            if before:
                return module.register_forward_pre_hook(lambda _module, args: tap(name, args[0]))
            return module.register_forward_hook(lambda _module, _args, value: tap(name, value))

        handles = [hook(name, *modules[name]) for name in POINTS]
        try:
            out = model(current, past_key_values=cache, use_cache=True, logits_to_keep=1)
            assert out.past_key_values is cache and all(x.get_seq_length() == pos + 1 for x in cache.layers)
            scored = scores(out.logits, (riley_id, other))
        finally:
            for handle in handles:
                handle.remove()
        cache.crop(-1)
        assert all(x.get_seq_length() == pos for x in cache.layers)
        assert tuple(effective) == POINTS
        assert summed_residual(effective["attention_residual"], effective["mlp_down"], emitted["layer_residual"])
        return emitted, effective, scored

    initial, _, control_logits = trial({})
    assert control_logits["logits_sha256"] == baseline["engines"]["hf"]["logits_sha256"]
    assert control_logits["lowest_max_token_id"] == other
    assert initial["attention_residual"] == hf_trace["attention_residual", "0"]
    assert initial["layer_residual"] == hf_trace["layer_residual", "0"]
    assert trial({key: initial[key] for key in POINTS[:-1]})[2] == control_logits

    settings = {
        "attention_context": ("attention_context",),
        "attention_projection": ("attention_projection",),
        "same_attention_residual": ("attention_residual",),
        "same_post_attention_norm": ("attention_residual", "post_attention_norm"),
        "same_gated_mlp_input": ("attention_residual", "post_attention_norm", "mlp_gated"),
        "same_mlp_down": ("attention_residual", "post_attention_norm", "mlp_gated", "mlp_down"),
    }
    output = {}
    for setting, points in settings.items():
        emitted, effective, scored = trial({name: riley[name, "0"] for name in points})
        if setting != "attention_context":
            assert effective["attention_residual"] == riley["attention_residual", "0"]
        if setting == "same_mlp_down":
            assert emitted["layer_residual"] == riley["layer_residual", "0"]
        output[setting] = {
            **scored,
            "operators_vs_riley": {name: {
                **difference(emitted[name], riley[name, "0"]),
                "emitted_sha256": digest(emitted[name]), "effective_sha256": digest(effective[name]),
                "riley_sha256": digest(riley[name, "0"]),
            } for name in POINTS},
        }
    assert trial({})[2] == control_logits and prefix_digest(cache, pos) == cache_hash
    return {
        "riley_token": riley_id, "hf_token": other, "position": pos, "input_token": token,
        "native_source_sha256": source_hashes, "hf_prefix_cache_sha256": cache_hash,
        "hf_baseline": control_logits, "trials": output,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("stage", "operator", "progression", "previous", "model"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    reference = prior_report(args.previous)
    assert digest((args.progression / "verified.json").read_bytes()) == BOUNDARY_TRACE_SHA256
    assert digest((args.operator / "verified.json").read_bytes()) == OPERATOR_TRACE_SHA256
    assert os.environ["CUDA_VISIBLE_DEVICES"] == GPU and torch.cuda.device_count() == 1
    assert torch.__version__ == reference["hf_torch"] and transformers.__version__ == reference["hf_transformers"]
    config = checkpoint(args.model)
    assert config["intermediate_size"] == INTERMEDIATE
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    assert len(model.model.layers) == LAYERS
    cases = {}
    with torch.inference_mode():
        for _, original, _, offset, other in targets(args.stage / "fixtures").values():
            cases[stem(original, offset)] = replay_case(
                model, args.stage, args.operator, args.progression, reference, original, offset, other)
    print(json.dumps({
        "checkpoint_manifest_sha256": CHECKPOINT, "gpu_uuid": GPU,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "prior_residual_summary_sha256": BOUNDARY_TRACE_SHA256,
        "prior_operator_summary_sha256": OPERATOR_TRACE_SHA256, "cases": cases,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
