"""Recompute the pinned HF decode steps with one Riley residual boundary at a time."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from capture_final_rows import CHECKPOINT, GPU, HIDDEN, VOCAB, checkpoint, row_bytes, stem, targets
from trace_decode_layers import LAYERS, check_endpoint, digest, position, prior_report, read_native, read_payload


ADDRESSABLE = 151_665
TRACE_SUMMARY_SHA256 = "396e59c9e9c34d56d34bf257679b49ee5a1108804a0f664fd056a93a86c89a06"
BOUNDARIES = [(name, str(layer)) for layer in range(LAYERS)
              for name in ("attention_residual", "layer_residual")]


def prefix_digest(cache, length):
    assert len(cache.layers) == LAYERS
    hasher = sha256()
    for layer in cache.layers:
        assert type(layer).__name__ == "DynamicLayer" and layer.get_seq_length() == length
        for tensor in (layer.keys, layer.values):
            assert tensor.dtype == torch.bfloat16 and tensor.shape[-2] == length
            hasher.update(tensor.contiguous().cpu().view(torch.int16).numpy().tobytes())
    return hasher.hexdigest()


def scores(logits, compared):
    raw = row_bytes(logits, VOCAB)
    candidate = logits[0, 0, :ADDRESSABLE].float()
    assert torch.isfinite(candidate).all().item()
    result = {}
    for token in compared:
        score = candidate[token]
        result[str(token)] = {
            "bf16_score": float(score.item()),
            "rank": 1 + int((candidate > score).sum().item()),
            "tied_count": int((candidate == score).sum().item()),
        }
    return {"logits_sha256": digest(raw), "lowest_max_token_id": int(candidate.argmax().item()),
            "scores": result}


def replay_case(model, trace, report, reference, label, original, offset, other):
    name = stem(original, offset)
    before_raw, riley = read_native(trace / "riley/rows" / name, original, offset)
    after_raw = (trace / "hf/rows" / f"{name}.bf16").read_bytes()
    hf = read_payload(after_raw)
    baseline = reference["cases"][name]
    source_hashes = {"riley": digest(before_raw), "hf": digest(after_raw)}
    pos, token = position(original, offset)
    trace_case = report["cases"][name]
    assert source_hashes == trace_case["source_trace_sha256"]
    assert (pos, token) == (trace_case["position"], trace_case["input_token"])
    check_endpoint(riley, baseline["engines"]["riley"], f"{label} Riley")
    check_endpoint(hf, baseline["engines"]["hf"], f"{label} HF")
    riley_id = original["tokens"][offset]
    assert (riley_id, other) == (baseline["riley_token"], baseline["hf_token"])
    compared = (riley_id, other)

    output = model(torch.tensor([original["prompt_tokens"]], device="cuda"),
                   use_cache=True, logits_to_keep=1)
    for prior in original["tokens"][:offset - 1]:
        output = model(torch.tensor([[prior]], device="cuda"),
                       past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    del output
    assert cache.get_seq_length() == pos
    cache_before = prefix_digest(cache, pos)
    current = torch.tensor([[token]], device="cuda")

    def decode():
        assert cache.get_seq_length() == pos
        output = model(current, past_key_values=cache, use_cache=True, logits_to_keep=1)
        assert output.past_key_values is cache
        assert all(layer.get_seq_length() == pos + 1 for layer in cache.layers)
        result = scores(output.logits, compared)
        cache.crop(-1)
        assert all(layer.get_seq_length() == pos for layer in cache.layers)
        return result

    initial = decode()
    assert initial["logits_sha256"] == baseline["engines"]["hf"]["logits_sha256"]
    assert initial["lowest_max_token_id"] == other
    assert {key: row["bf16_score"] for key, row in initial["scores"].items()} == baseline["engines"]["hf"]["observed_scores"]

    def replace(boundary, row):
        stage, layer_id = boundary
        layer = model.model.layers[int(layer_id)]
        tensor = torch.frombuffer(bytearray(row), dtype=torch.bfloat16).reshape(1, 1, HIDDEN).cuda()
        hits = 0

        def assign(value):
            nonlocal hits
            assert value.shape == (1, 1, HIDDEN) and value.dtype == torch.bfloat16
            assert row_bytes(value, HIDDEN) == hf[boundary]
            value.copy_(tensor)
            assert row_bytes(value, HIDDEN) == row
            hits += 1

        if stage == "attention_residual":
            # Qwen2 retains this same tensor for the later residual addition.
            hook = layer.post_attention_layernorm.register_forward_pre_hook(
                lambda _module, args: assign(args[0]))
        else:
            hook = layer.register_forward_hook(lambda _module, _args, result: assign(result))
        try:
            result = decode()
        finally:
            hook.remove()
        assert hits == 1
        return result

    replacements = []
    for boundary in BOUNDARIES:
        control = replace(boundary, hf[boundary])
        assert control == initial, f"HF self-replacement changed logits at {boundary}"
        trial = replace(boundary, riley[boundary])
        replacements.append({"stage": boundary[0], "layer": int(boundary[1]),
                             "riley_residual_sha256": digest(riley[boundary]), **trial})
    assert decode() == initial, "unhooked HF repeat differs after replacements"
    assert prefix_digest(cache, pos) == cache_before
    terminal = replacements[-1]
    terminal_scores = {key: row["bf16_score"] for key, row in terminal["scores"].items()}
    return {
        "position": pos, "input_token": token, "riley_token": riley_id, "hf_token": other,
        "source_trace_sha256": source_hashes, "hf_prefix_cache_sha256": cache_before,
        "hf_baseline": initial, "exact_self_replacement_controls": len(replacements),
        "replacement_trials": replacements,
        "terminal_replacement_matches_riley_scores": terminal_scores == baseline["engines"]["riley"]["observed_scores"],
        "terminal_replacement_matches_riley_logit_hash": terminal["logits_sha256"] == baseline["engines"]["riley"]["logits_sha256"],
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    ref = prior_report(args.reference)
    trace_json = (args.trace / "verified.json").read_bytes()
    assert digest(trace_json) == TRACE_SUMMARY_SHA256
    report = json.loads(trace_json)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == GPU and torch.cuda.device_count() == 1
    assert torch.__version__ == ref["hf_torch"] and transformers.__version__ == ref["hf_transformers"]
    checkpoint(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    assert len(model.model.layers) == LAYERS
    cases = {}
    with torch.inference_mode():
        for label, (_, original, _, offset, other) in targets(args.trace / "fixtures").items():
            cases[stem(original, offset)] = replay_case(model, args.trace, report, ref, label, original, offset, other)
    print(json.dumps({
        "checkpoint_manifest_sha256": CHECKPOINT, "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu_uuid": GPU, "addressable_vocabulary": ADDRESSABLE,
        "layer_trace_summary_sha256": TRACE_SUMMARY_SHA256, "cases": cases,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
