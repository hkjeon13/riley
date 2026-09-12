import os,json,pathlib,hashlib
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np,torch
from transformers import AutoModelForCausalLM
r=pathlib.Path('/tmp/riley-opt-260912/shared-natural-v10');policy=json.loads((r/'policy.json').read_text());corpus=json.loads((r/'corpus.json').read_text());assert hashlib.sha256((r/'corpus.json').read_bytes()).hexdigest()==policy['corpus_sha256']
torch.set_num_threads(8);m=AutoModelForCausalLM.from_pretrained('/data/riley-benchmark/20260827T051948Z-d7ad713a/model',torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True).eval();records=[];documents=[]
def read(p):
 bits=np.fromfile(p,dtype='<u2').astype(np.uint32)<<16;assert bits.size==49152
 return torch.from_numpy(bits.view(np.float32).astype(np.float64))
with torch.inference_mode():
 for case in range(8):
  base=r/'dumps'/f'batch{case}';cache={};seen={};rows=[json.loads(x) for x in (base/'rows.jsonl').read_text().splitlines()];assert len(rows)==256
  for row in rows:
   request=row['request'];doc=case*8+request-1;tokens=corpus[doc]['tokens'];length=row['target_length'];provided=row['input_tokens'];expected=tokens[:128] if length==128 else [tokens[length-1]];assert provided==expected
   result=m(input_ids=torch.tensor([provided]),past_key_values=cache.get(request),use_cache=True);cache[request]=result.past_key_values;assert cache[request].get_seq_length()==length
   ref=result.logits[0,-1].double();lr=ref.log_softmax(-1);prob=lr.exp();label=tokens[length]
   a=read(base/(row['name']+'-original.bin'));b=read(base/(row['name']+'-shared.bin'));assert bool(a.isfinite().all() and b.isfinite().all() and ref.isfinite().all())
   x={'document':doc,'length':length,'label':label,'top1_fp32':int(ref.argmax()),'fp32_nll':float(-lr[label])}
   for name,values in [('original',a),('shared',b)]:
    logp=values.log_softmax(-1);x[name+'_nll']=float(-logp[label]);x[name+'_kl']=float((prob*(lr-logp)).sum());x[name+'_top1']=int(values.argmax())
   records.append(x);seen.setdefault(doc,[]).append(x)
  for doc,xs in seen.items():
   assert len(xs)==32 and sorted(x['length'] for x in xs)==list(range(128,160))
   documents.append({'document':doc,'delta_nll':float(np.mean([x['shared_nll']-x['original_nll'] for x in xs]))})
  print('analyzed batch',case,flush=True)
assert len(records)==2048 and len(documents)==64
values=np.array([x['delta_nll'] for x in documents]);rng=np.random.default_rng(policy['bootstrap_seed']);boot=values[rng.integers(0,64,size=(policy['bootstrap_repetitions'],64))].mean(axis=1);upper=float(np.quantile(boot,.95));summary={'rows':len(records),'passages':len(documents),'mean_delta_nll_nats':float(values.mean()),'upper95_bootstrap_mean_delta_nll':upper,'max_passage_mean_delta_nll':float(values.max()),'full_quality_qualified':False,'confidence_scope':'resampling64text windows; does not guarantee article independence or generalization'}
for name in ['fp32','original','shared']:
 summary[name+'_mean_nll']=float(np.mean([x[name+'_nll'] for x in records]))
 if name!='fp32':
  kl=[x[name+'_kl'] for x in records];summary[name+'_mean_kl']=float(np.mean(kl));summary[name+'_p99_kl']=float(np.quantile(kl,.99));summary[name+'_top1_matches_fp32']=sum(x[name+'_top1']==x['top1_fp32'] for x in records)
summary['shared_top1_matches_original']=sum(x['shared_top1']==x['original_top1'] for x in records)
summary['checks']={'mean_nll_ci':upper<=policy['mean_nll_delta_upper_95_ci_nats'],'passage_nll':float(values.max())<=policy['max_document_mean_nll_delta_nats'],'mean_kl':summary['shared_mean_kl']<=summary['original_mean_kl']+policy['mean_kl_from_fp32_additive_slack'],'p99_kl':summary['shared_p99_kl']<=summary['original_p99_kl']+policy['p99_kl_from_fp32_additive_slack'],'finite':True};summary['finite_corpus_screen_pass']=all(summary['checks'].values())
for n,x in [('fp32-rows.json',records),('document-deltas.json',documents),('summary.json',summary)]:
 with (r/n).open('x') as f:json.dump(x,f,indent=2)
print(json.dumps(summary),flush=True)
