import os,json,pathlib,math
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np,torch
from transformers import AutoModelForCausalLM
r=pathlib.Path('/tmp/riley-opt-260912/shared-model-diagnostic-v10-r1')
rows=[json.loads(s) for s in (r/'rows.jsonl').read_text().splitlines()]
policy={'scope':'paired diagnostic flags only, not model/serving acceptance','rmse_ratio':1.1,'rmse_abs_slack':0.0001,'kl_ratio':1.1,'kl_abs_slack':0.00001,'inputs':'same native original-policy teacher-forced histories; eight synthetic P128 inputs and mixed output lengths'}
with (r/'fp32-policy.json').open('x') as f:json.dump(policy,f,indent=2)
torch.set_num_threads(8)
m=AutoModelForCausalLM.from_pretrained('/data/riley-benchmark/20260827T051948Z-d7ad713a/model',torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True).eval();cache={};results=[]
def read(name):
 bits=np.fromfile(r/name,dtype='<u2').astype(np.uint32)<<16
 assert bits.size==49152
 return torch.from_numpy(bits.view(np.float32).astype(np.float64))
with torch.inference_mode():
 for row in rows:
  req=row['request'];tokens=row['input_tokens'];out=m(input_ids=torch.tensor([tokens]),past_key_values=cache.get(req),use_cache=True);cache[req]=out.past_key_values
  assert cache[req].get_seq_length()==row['target_length']
  ref=out.logits[0,-1].double();lr=ref.log_softmax(-1);prob=lr.exp();a=read(row['name']+'-original.bin');b=read(row['name']+'-shared.bin')
  z={'name':row['name'],'request':req,'length':row['target_length'],'top_original':int(a.argmax()),'top_shared':int(b.argmax()),'top_fp32':int(ref.argmax())}
  for key,x in [('original',a),('shared',b)]:
   assert bool(x.isfinite().all());z[key+'_rmse']=float(((x-ref).square().mean()).sqrt());z[key+'_max_abs']=float((x-ref).abs().max());z[key+'_kl_from_fp32']=float((prob*(lr-x.log_softmax(-1))).sum())
  z['rmse_flag']=z['shared_rmse']>z['original_rmse']*policy['rmse_ratio']+policy['rmse_abs_slack'];z['kl_flag']=z['shared_kl_from_fp32']>z['original_kl_from_fp32']*policy['kl_ratio']+policy['kl_abs_slack'];results.append(z)
summary={'rows':len(results),'original_top1_fp32':sum(x['top_original']==x['top_fp32'] for x in results),'shared_top1_fp32':sum(x['top_shared']==x['top_fp32'] for x in results),'shared_top1_original':sum(x['top_shared']==x['top_original'] for x in results),'rmse_flags':sum(x['rmse_flag'] for x in results),'kl_flags':sum(x['kl_flag'] for x in results),'quality_qualified':False}
for prefix in ['original','shared']:
 for metric in ['rmse','max_abs','kl_from_fp32']:
  xs=[x[prefix+'_'+metric] for x in results];summary[prefix+'_'+metric]={'mean':sum(xs)/len(xs),'max':max(xs)}
(r/'fp32-rows.json').write_text(json.dumps(results,indent=2)+'\n');(r/'fp32-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
