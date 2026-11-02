#!/usr/bin/env python3
"""Correctness of long prompts: concurrency invariance and agreement with HF.

usage: long_check.py ADDRESS MODEL_ID MODEL_DIR OUT
Greedy continuations of documents of about 800 to 7000 tokens are requested
alone, then at 8 and 32 concurrent clients, and compared with each other and
with Hugging Face BF16 sdpa.
"""
import concurrent.futures as futures
import json
import random
import sys
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

address, model_id, model_dir, out = sys.argv[1:5]
words = ("the river market station garden winter engine letter mountain copper signal harbor forest planet "
         "window silver thread summer lantern village theory circuit pattern shadow bridge ocean").split()
rng = random.Random(5)
prompts = []
for length in [600, 1200, 2000, 3000, 4000, 5500] * 3:
    text = " ".join(rng.choice(words) + ("." if rng.random() < .1 else "") for _ in range(length))
    prompts.append(text + "\n\nIn short, the document is about")


def complete(prompt):
    body = json.dumps({"model": model_id, "prompt": prompt, "max_tokens": 32, "temperature": 0,
                       "return_token_ids": True}).encode()
    request = urllib.request.Request(f"http://{address}/v1/completions", body, {"Content-Type": "application/json"})
    choice = json.load(urllib.request.urlopen(request, timeout=600))["choices"][0]
    return choice["token_ids"], choice["prompt_token_ids"]


alone = [complete(prompt) for prompt in prompts]
result = {"prompts": len(prompts), "prompt_tokens": [len(item[1]) for item in alone]}
for clients in (8, 32):
    with futures.ThreadPoolExecutor(clients) as pool:
        again = list(pool.map(complete, prompts * 3))
    result[f"invariant_c{clients}"] = sum(again[i][0] == alone[i % len(prompts)][0] for i in range(len(again)))
    result[f"of_c{clients}"] = len(again)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
same, firsts, depth = 0, 0, []
for generated, prompt_ids in alone:
    ids = torch.tensor([prompt_ids]).cuda()
    with torch.no_grad():
        output = model.generate(ids, max_new_tokens=32, do_sample=False, temperature=None, top_p=None, top_k=None,
                                repetition_penalty=None, pad_token_id=tokenizer.eos_token_id)
    reference = output[0, len(prompt_ids):].tolist()
    if reference[-1] in (tokenizer.eos_token_id,):
        reference = reference[:-1]
    candidate = [token for token in generated if token != tokenizer.eos_token_id]
    same += reference == candidate
    firsts += bool(reference and candidate and reference[0] == candidate[0])
    depth.append(next((i for i, (a, b) in enumerate(zip(reference, candidate)) if a != b), len(candidate)))
result.update(hf_identical=same, hf_first_token=firsts, hf_common_prefix=depth)
print(json.dumps({k: v for k, v in result.items() if k != "prompt_tokens"}))
json.dump(result, open(out, "w"), indent=1)
