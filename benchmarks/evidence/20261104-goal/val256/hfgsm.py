#!/usr/bin/env python3
"""hfgsm.py MODEL_DIR DATA_DIR OUT.json: GSM8K with HF Transformers greedy (BF16, eager attention), the prompts and scoring of gsm8k.py; batches of 16, left padding."""
import json, re, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m, data, out = sys.argv[1:4]
rows = lambda s: [json.loads(l) for l in open(f"{data}/{s}.jsonl")]
def number(t):
    f = re.findall(r"-?\d[\d,]*\.?\d*", t); return f[-1].replace(",", "").rstrip(".") if f else None
shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in rows("train")[:4])
test = rows("test")
tok = AutoTokenizer.from_pretrained(m); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
res = []
for i in range(0, len(test), 16):
    b = test[i:i+16]; enc = tok([shots + f"Question: {r['question']}\nAnswer:" for r in b], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad(): o = model.generate(**enc, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
    for r, seq in zip(b, o):
        t = tok.decode(seq[enc.input_ids.shape[1]:], skip_special_tokens=True).split("Question:")[0]
        g = r["answer"].split("####")[-1].strip().replace(",", ""); p = number(t.split("####")[-1]) if "####" in t else number(t)
        try: c = p is not None and float(p) == float(g)
        except ValueError: c = False
        res.append({"correct": c, "text": t})
    print(i, flush=True)
acc = sum(r["correct"] for r in res) / len(res)
json.dump({"accuracy": acc, "questions": len(res), "results": res}, open(out, "w"))
print(json.dumps({"accuracy": round(100 * acc, 2), "questions": len(res)}))
