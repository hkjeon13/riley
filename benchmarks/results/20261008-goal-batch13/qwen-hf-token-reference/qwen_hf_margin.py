"""For each prompt where the Riley scheduler route first differs from HF greedy, report HF's logit margin at that position:
top-1 vs top-2 logit (float32 view of the BF16 model output) and the rank of Riley's token. Teacher-forced on the common prefix."""
import json, sys, torch, transformers
R = '/data/riley-serving-261007'; O = f'{R}/qwen-hf-token-reference-attempt01'
model = '/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1'
fx = json.load(open(f'{R}/kernel-batch13-controller-attempt01/fixtures.json')); items = {(w, c['id']): c for w in ('fixed', 'natural') for c in fx[w]['corpus']}
res = json.load(open(f'{O}/result.json')); hf = json.load(open(f'{O}/hf.json')); ref = {(r['workload'], r['id']): r['tokens'] for r in hf['rows']}
tok = transformers.AutoTokenizer.from_pretrained(model); m = transformers.AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, attn_implementation='eager').cuda().eval()
out = []
for row in res['runs'][0]['rows']:
    d = row['first_difference']
    if d is None: continue
    key = (row['workload'], row['id']); ids = tok(items[key]['prompt'], return_tensors='pt', add_special_tokens=False).input_ids
    full = torch.cat([ids, torch.tensor([ref[key][:d]], dtype=torch.long)], 1).cuda()
    with torch.no_grad(): logits = m(full).logits[0, -1].float()
    top = torch.topk(logits, 5); riley_token = row['riley_tokens'][d] if d < len(row['riley_tokens']) else None
    rank = None if riley_token is None else int((logits > logits[riley_token]).sum())
    out.append({'workload': key[0], 'id': key[1], 'index': d, 'hf_token': ref[key][d] if d < len(ref[key]) else None, 'riley_token': riley_token, 'hf_top5_ids': top.indices.tolist(), 'hf_top5_logits': top.values.tolist(),
                'top1_minus_top2': float(top.values[0] - top.values[1]), 'riley_token_rank_in_hf': rank, 'riley_token_logit_gap': None if riley_token is None else float(top.values[0] - logits[riley_token]),
                'bf16_step_at_top1': float(torch.tensor(top.values[0].item(), dtype=torch.bfloat16).float().abs() * 2 ** -8)})
    print(out[-1], flush=True)
json.dump(out, open(f'{O}/hf-margins.json', 'w'), indent=1)
