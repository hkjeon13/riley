#!/usr/bin/env python3
"""Compare Riley's logprobs with Hugging Face BF16 eager on the same token path.

usage: logprob_oracle.py ADDRESS MODEL_ID MODEL_DIR GPU [--out FILE]
Riley generates greedily; HF replays exactly those tokens, so a difference is a
difference in numbers, not in which token was chosen.
"""
import argparse
import json
import math
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("address")
parser.add_argument("model_id")
parser.add_argument("model_dir")
parser.add_argument("--out")
args = parser.parse_args()

PROMPTS = ["The capital of France is", "Write a haiku about rivers.", "List three prime numbers.",
           "def fibonacci(n):", "Explain gravity in one sentence.", "Translate 'good morning' to Spanish.",
           "Water boils at", "The opposite of hot is"]
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").cuda().eval()
differences, overlaps, tokens = [], [], 0
for prompt in PROMPTS:
    body = json.dumps({"model": args.model_id, "prompt": prompt, "max_tokens": 32, "temperature": 0,
                       "logprobs": 5, "return_token_ids": True}).encode()
    request = urllib.request.Request(f"http://{args.address}/v1/completions", body,
                                     {"Content-Type": "application/json"})
    choice = json.load(urllib.request.urlopen(request, timeout=120))["choices"][0]
    generated, legacy = choice["token_ids"], choice["logprobs"]
    prompt_ids = choice.get("prompt_token_ids") or tokenizer(prompt).input_ids
    ids = torch.tensor([prompt_ids + generated]).cuda()
    with torch.no_grad():
        logits = model(ids).logits[0, len(prompt_ids) - 1:-1].float()
    reference = torch.log_softmax(logits, dim=-1)
    for position, token in enumerate(generated):
        tokens += 1
        differences.append(abs(reference[position, token].item() - legacy["token_logprobs"][position]))
        top = reference[position].topk(5).indices.tolist()
        shown = {tokenizer.decode([index]) for index in top}
        overlaps.append(len(shown & set(legacy["top_logprobs"][position])) / 5)
differences.sort()
result = {"tokens": tokens, "abs_diff_p50": differences[len(differences) // 2],
          "abs_diff_p99": differences[int(len(differences) * .99)], "abs_diff_max": differences[-1],
          "top5_overlap_mean": sum(overlaps) / len(overlaps)}
print(json.dumps(result))
if args.out:
    json.dump(result, open(args.out, "w"), indent=1)
assert math.isfinite(result["abs_diff_max"])
