import os
os.environ['RILEY_MMA_LIBRARY']='mma_attention_graph.so'
import torch,json
from pathlib import Path
from mma_probe import native
from vllm.vllm_flash_attn import flash_attn_varlen_func
out=Path('/tmp/riley-g04-attention-policy-260911')
torch.manual_seed(1701);records=[]
with torch.inference_mode():
 for n,rows in [(1,1),(17,1),(17,17),(64,64),(65,1),(65,65),(128,1),(128,128),(129,1),(129,17),(160,1),(160,32)]:
  q=torch.randn((rows,9,64),device='cuda',dtype=torch.bfloat16)
  k=torch.randn((n,3,64),device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
  blocks=(n+15)//16;kp=torch.zeros((blocks*16,3,64),device='cuda',dtype=torch.bfloat16);vp=torch.zeros_like(kp);kp[:n]=k;vp[:n]=v
  expected=flash_attn_varlen_func(q=q,k=kp.reshape(blocks,16,3,64),v=vp.reshape(blocks,16,3,64),cu_seqlens_q=torch.tensor([0,rows],device='cuda',dtype=torch.int32),max_seqlen_q=rows,seqused_k=torch.tensor([n],device='cuda',dtype=torch.int32),max_seqlen_k=n,softmax_scale=.125,causal=True,block_table=torch.arange(blocks,device='cuda',dtype=torch.int32)[None,:],fa_version=2,num_splits=0)
  if rows!=1 and rows!=n:
   try: native(q,k,v)
   except AssertionError: records.append({'sequence_length':n,'query_rows':rows,'unsupported_rejected':True});continue
   raise AssertionError('unsupported mixed prefill accepted')
  actual=native(q,k,v)
  records.append({'sequence_length':n,'query_rows':rows,'unequal':int((actual!=expected).sum().item()),'max_abs':float((actual.float()-expected.float()).abs().max().item())})
(out/'random-comparison.json').write_text(json.dumps({'seed':1701,'records':records,'performance_trials':0},indent=2)+'\n')
print(records)
