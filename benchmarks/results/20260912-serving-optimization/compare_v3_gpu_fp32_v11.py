import os,pathlib,struct,json
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np,torch
from transformers import AutoModelForCausalLM
r=pathlib.Path('/tmp/riley-opt-260912/prefill-full-model-v11');torch.set_num_threads(8)
m=AutoModelForCausalLM.from_pretrained('/data/riley-benchmark/20260827T051948Z-d7ad713a/model',torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True).eval()
records=[]
with torch.inference_mode(),(r/'requests.bin').open('rb') as f:
 count=struct.unpack('<I',f.read(4))[0]
 for _ in range(count):
  n=struct.unpack('<I',f.read(4))[0];ids=list(struct.unpack('<'+'I'*n,f.read(n*4)))
  ref=m.model(input_ids=torch.tensor([ids]),use_cache=False).last_hidden_state[0,-1]
  bits=np.fromfile(r/f'hidden-{n}.bf16',dtype='<u2').astype(np.uint32)<<16;assert bits.size==576
  actual=torch.from_numpy(bits.view(np.float32));assert bool(actual.isfinite().all())
  gpu_bits=np.fromfile(r/f'rust-logits-{n}.bf16',dtype='<u2').astype(np.uint32)<<16;assert gpu_bits.size==49152
  a=torch.from_numpy(gpu_bits.view(np.float32)).double();assert bool(a.isfinite().all());b=m.lm_head(ref).double();la=a.log_softmax(-1);lb=b.log_softmax(-1)
  records.append({'prompt_tokens':n,'hidden_rmse':float((actual-ref).square().mean().sqrt()),'hidden_max_abs':float((actual-ref).abs().max()),'gpu_logits_kl_from_fp32':float((lb.exp()*(lb-la)).sum()),'gpu_top1':int(a.argmax()),'fp32_top1':int(b.argmax()),'finite':True})
x={'scope':'three checkpoint actual GPU logits versus CPU FP32 diagnostics; not general quality acceptance','records':records,'quality_qualified':False}
(r/'rust-gpu-fp32-diagnostic.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
