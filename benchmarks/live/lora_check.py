#!/usr/bin/env python3
"""Make a random LoRA adapter, then give the PEFT merge as the reference.

usage: lora_check.py make MODEL_DIR ADAPTER_DIR
       lora_check.py compare MODEL_DIR ADAPTER_DIR MERGED_DIR ADDRESS MODEL_ID OUT
`compare` checks the merged weights against PEFT's merge_and_unload and the
served greedy text against PEFT's generation on the same prompts.
"""
import json
import sys
import urllib.request

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob

PROMPTS = ["The capital of France is", "List three prime numbers.", "Explain gravity in one sentence.",
           "def add(a, b):", "Water boils at", "Name a color."]
mode, model_dir, adapter = sys.argv[1:4]
torch.manual_seed(3)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
if mode == "make":
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj", "gate_proj"],
                        init_lora_weights=False)
    peft = get_peft_model(model, config)
    for name, parameter in peft.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parameter.data = torch.randn_like(parameter.data.float()) * 0.02
    peft.save_pretrained(adapter)
    raise SystemExit(0)

merged_dir, address, model_id, out = sys.argv[4:8]
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# `compare-base` measures the same agreement without any adapter, the baseline
# for how often served text and HF text agree at all.
reference = (model if mode == "compare-base"
             else PeftModel.from_pretrained(model, adapter).merge_and_unload()).eval()
state = reference.state_dict()
bits, total, largest, tensors = 0, 0, 0.0, 0
for path in glob.glob(f"{merged_dir}/*.safetensors"):
    with safe_open(path, "pt") as handle:
        for name in handle.keys():
            if name not in state:
                continue
            ours = handle.get_tensor(name).to(state[name].device)
            tensors += 1
            differing = (ours.view(-1).float() != state[name].view(-1).float())
            bits += differing.sum().item()
            total += ours.numel()
            if differing.any():
                largest = max(largest, (ours.float() - state[name].float()).abs().max().item())
texts, equal = [], 0
for prompt in PROMPTS:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output = reference.generate(ids, max_new_tokens=24, do_sample=False, temperature=None, top_p=None,
                                    top_k=None, repetition_penalty=None, pad_token_id=tokenizer.eos_token_id)
    expected = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
    body = json.dumps({"model": model_id, "prompt": prompt, "max_tokens": 24, "temperature": 0}).encode()
    request = urllib.request.Request(f"http://{address}/v1/completions", body, {"Content-Type": "application/json"})
    served = json.load(urllib.request.urlopen(request, timeout=120))["choices"][0]["text"]
    equal += served == expected
    texts.append({"prompt": prompt, "served": served, "peft": expected})
result = {"tensors": tensors, "values": total, "differing_values": bits, "largest_difference": largest,
          "prompts": len(PROMPTS), "identical_text": equal, "texts": texts}
print(json.dumps({key: value for key, value in result.items() if key != "texts"}))
json.dump(result, open(out, "w"), indent=1, ensure_ascii=False)
