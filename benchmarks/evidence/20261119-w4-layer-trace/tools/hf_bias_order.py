"""Measure the effect of rounding QKV before adding bias in the HF SDPA reference."""

import argparse
from hashlib import sha256
from pathlib import Path

import torch
from torch.nn.functional import linear
from transformers import AutoModelForCausalLM

from layer_trace import pinned_case
from long_accuracy import save


def split_bias(projection, args, _output):
    return linear(args[0], projection.weight, None) + projection.bias


def measure(model, tokens, site):
    layers = model.model.layers
    if site == "all_qkv":
        modules = [head for layer in layers for head in
                   (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj)]
    elif site == "layer0_qkv":
        layer = layers[0].self_attn
        modules = [layer.q_proj, layer.k_proj, layer.v_proj]
    elif site == "layer0_q":
        modules = [layers[0].self_attn.q_proj]
    else:
        modules = []
    hooks = []
    try:
        for module in modules:
            if module.bias is None:
                raise ValueError("a QKV projection is missing its bias")
            hooks.append(module.register_forward_hook(split_bias))
        with torch.inference_mode():
            logits = model(tokens, use_cache=True, logits_to_keep=1).logits[0, 0].cpu()
    finally:
        for hook in hooks:
            hook.remove()
    scores, ids = logits.float().topk(5)
    return {
        "first_id": int(logits.argmax()),
        "top_5": [{"id": token, "logit": score} for token, score in zip(ids.tolist(), scores.tolist())],
        "tracked_logits": {str(token): float(logits[token]) for token in (264, 5257)},
        "logits_sha256": sha256(logits.view(torch.uint16).numpy().tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    results = {}
    for case, expected in ((3, 5257), (9, 264)):
        row, _ = pinned_case(args.fixture, case)
        tokens = torch.tensor([row["prompt_tokens"]], device="cuda")
        runs = {site: measure(model, tokens, site)
                for site in ("baseline", "layer0_q", "layer0_qkv", "all_qkv", "baseline_repeat")}
        if runs["baseline"] != runs["baseline_repeat"] or runs["baseline"]["first_id"] != expected:
            raise ValueError(f"the baseline failed to reproduce the historical HF reference for case {case}")
        results[str(case)] = {"prompt_tokens": tokens.numel(), "prompt_sha256": row["prompt_sha256"],
                              "riley_first_id": row["tokens"][0], "runs": runs}
    save(args.output, {"torch": torch.__version__, "backend": "sdpa", "cases": results})


if __name__ == "__main__":
    main()
