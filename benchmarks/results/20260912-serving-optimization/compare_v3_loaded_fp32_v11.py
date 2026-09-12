import os,pathlib,struct,json
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from transformers import AutoModelForCausalLM
r=pathlib.Path('/tmp/riley-opt-260912/loaded-rope-fixture-v11');torch.set_num_threads(8)
m=AutoModelForCausalLM.from_pretrained('/data/riley-benchmark/20260827T051948Z-d7ad713a/model',torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True).eval();rows=[]
with (r/'requests.bin').open('rb') as f,torch.inference_mode():
 cases=struct.unpack('<I',f.read(4))[0]
 for _ in range(cases):
  n=struct.unpack('<I',f.read(4))[0];prompt=list(struct.unpack('<'+'I'*n,f.read(n*4)));tokens=json.loads((r/f'decode-tokens-{n}.json').read_text());bits=np.fromfile(r/f'decode-logits-{n}.bf16',dtype='<u2').astype(np.uint32)<<16;logits=bits.view(np.float32).reshape(len(tokens),49152);cache=None
  for i,token in enumerate(tokens):
   out=m(input_ids=torch.tensor([prompt if i==0 else [tokens[i-1]]]),past_key_values=cache,use_cache=True);cache=out.past_key_values;assert cache.get_seq_length()==n+i
   a=torch.from_numpy(logits[i]).double();b=out.logits[0,-1].double();assert bool(a.isfinite().all() and b.isfinite().all());assert int(a.argmax())==token
   la=a.log_softmax(-1);lb=b.log_softmax(-1);top=b.topk(2).values
   rows.append({'prompt':n,'generated_index':i,'token':token,'fp32_top1':int(b.argmax()),'kl':float((lb.exp()*(lb-la)).sum()),'fp32_top2_margin':float(top[0]-top[1])})
  print('compared',n,len(tokens),flush=True)
summary={'scope':'common generated history, not independent FP32 greedy runs or task quality acceptance','rows':len(rows),'top1_matches':sum(x['token']==x['fp32_top1'] for x in rows),'mean_kl':float(np.mean([x['kl'] for x in rows])),'p99_kl':float(np.quantile([x['kl'] for x in rows],.99)),'max_kl':max(x['kl'] for x in rows),'distinct_generated_tokens':len({x['token'] for x in rows}),'quality_qualified':False}
(r/'decode-fp32-rows.json').write_text(json.dumps(rows,indent=2)+'\n');(r/'decode-fp32-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
