"""Alter only the HF SDPA prefill partition on the two pinned trace prompts."""

import argparse
from hashlib import sha256
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from layer_trace import pinned_case
from long_accuracy import save


def measure(model, tokens, partition):
    offset, cache = 0, None
    with torch.inference_mode():
        for width in partition:
            result = model(tokens[:, offset:offset + width], past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            cache = result.past_key_values
            offset += width
            if cache.get_seq_length() != offset:
                raise ValueError(f"cache length differs from consumed prompt at {offset}")
        if offset != tokens.numel():
            raise ValueError("partition does not cover the full prompt")
        logits = result.logits[0, 0].cpu().contiguous()
    return {"partition": partition, "first_id": int(logits.argmax()),
            "tracked_logits": {str(token): float(logits[token]) for token in (264, 5257)},
            "logits_sha256": sha256(logits.view(torch.uint16).numpy().tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    cases = {}
    for case, expected, native_tail in ((3, 5257, 229), (9, 264, 2)):
        row, _ = pinned_case(args.fixture, case)
        tokens = torch.tensor([row["prompt_tokens"]], device="cuda")
        length = tokens.numel()
        prefix_chunks, pre_tail = divmod(length - native_tail, 1024)
        partition = [1024] * prefix_chunks + ([pre_tail] if pre_tail else []) + [native_tail]
        baselines = [length]
        arms = {"baseline": baselines, "native_final_tail": [length - native_tail, native_tail],
                "budget_1024_with_native_final_tail": partition, "baseline_repeat": baselines}
        runs = {name: measure(model, tokens, pieces) for name, pieces in arms.items()}
        if runs["baseline"] != runs["baseline_repeat"] or runs["baseline"]["first_id"] != expected:
            raise ValueError(f"the baseline failed to reproduce the historical HF reference for case {case}")
        cases[str(case)] = {"prompt_sha256": row["prompt_sha256"],
                           "riley_first_id": row["tokens"][0], "runs": runs}
    save(args.output, {"torch": torch.__version__, "backend": "sdpa", "cases": cases})


if __name__ == "__main__":
    main()
