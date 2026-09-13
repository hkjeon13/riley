#!/usr/bin/env python3
"""Offline-only FP32 observations at identical histories; not a quality gate."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

parser=argparse.ArgumentParser()
parser.add_argument('--model',required=True)
parser.add_argument('--generation-log',type=Path,required=True)
args=parser.parse_args()
records={}
for line in args.generation_log.read_text().splitlines():
    if line.startswith('FREE_TOKENS '):
        name,data=line.removeprefix('FREE_TOKENS ').split('=',1)
        records[name]=json.loads(data)
assert set(records)=={'baseline','candidate'}
assert all(len(v)==32 and all(len(row)==32 for row in v) for v in records.values())
state=9131701
prompts=[]
for length in [1,15,16,17,127,128,129,511]:
    prompt=[]
    for _ in range(length):
        state=(state*6364136223846793005+1442695040888963407)&((1<<64)-1)
        prompt.append(32+((state>>32)%49000))
    prompts.append(prompt)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,
    attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval().to('cuda')
print(json.dumps({'reference':'HF eager FP32, TF32 disabled','serving_integration':False,
                  'parameter_dtype':str(next(model.parameters()).dtype),'torch':torch.__version__}),flush=True)
with torch.inference_mode():
    for i,prompt in enumerate(prompts):
        a,b=records['baseline'][i],records['candidate'][i]
        first=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),None)
        if first is None:continue
        assert a[:first]==b[:first]
        input_ids=torch.tensor([prompt+a[:first]],device='cuda',dtype=torch.long)
        logits=model(input_ids=input_ids,use_cache=False).logits[0,-1].float()
        assert bool(torch.isfinite(logits).all())
        probabilities=logits.log_softmax(-1)
        values,ids=logits.topk(5)
        print(json.dumps({'prompt_index':i,'prompt_tokens':len(prompt),'first_index':first,
          'baseline_token':a[first],'candidate_token':b[first],'fp32_top_tokens':ids.tolist(),
          'fp32_top_logits':values.tolist(),'baseline_log_probability':float(probabilities[a[first]]),
          'candidate_log_probability':float(probabilities[b[first]]),
          'general_quality_accepted':False}),flush=True)
