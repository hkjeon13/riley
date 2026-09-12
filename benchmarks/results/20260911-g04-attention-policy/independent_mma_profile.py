"""Replay compiled model segments with actual serving attention outputs as inputs."""
import torch,json
from pathlib import Path
from safetensors.torch import load_file
r=Path('/tmp/riley-g04-serving-trace-260911');out=Path('/tmp/riley-g04-attention-policy-260911')
env=json.loads(Path('/tmp/riley-g04-readiness-260911/environment.json').read_text())
w=load_file(env['vllm_checkpoint']+'/model.safetensors',device='cuda')
ref=json.loads((r/'invariance.json').read_text())['before']
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
# vLLM casts its rotary cache to the query BF16 dtype before rotation.
freq=1.0/(100000.0**(torch.arange(0,64,2,device='cuda').float()/64))
angles=torch.arange(160,device='cuda').float()[:,None]*freq[None,:]
cos=angles.cos().bfloat16();sin=angles.sin().bfloat16()
def norm(x,weight):
 f=x.float();return (f*torch.rsqrt((f*f).mean(-1,keepdim=True)+1e-5)*weight.float()).bfloat16()
def prefix(h,res,nw,qkvw,c,s,first):
 z=h if first else h.float()+res.float()
 residual=z.bfloat16()
 q,k,v=torch.nn.functional.linear(norm(z,nw),qkvw).split([576,192,192],dim=-1)
 def rotate(x,heads):
  t=x.reshape(-1,heads,64);a,b=t.chunk(2,-1)
  return torch.cat((a*c[:,None,:]-b*s[:,None,:],b*c[:,None,:]+a*s[:,None,:]),-1)
 return rotate(q,9),rotate(k,3),v.reshape(-1,3,64),residual
def tail(context,res,ow,nw,guw,dw):
 p=torch.nn.functional.linear(context,ow)
 z=p.float()+res.float();res=z
 gu=torch.nn.functional.linear(norm(z,nw),guw);g,u=gu.chunk(2,-1)
 h=(torch.nn.functional.silu(g.float())*u.float()).bfloat16()
 return torch.nn.functional.linear(h,dw),res
cp=torch.compile(prefix,fullgraph=True,dynamic=True);ct=torch.compile(tail,fullgraph=True,dynamic=True)
def read(layer,call,name):
 return torch.frombuffer(bytearray((r/f'layer{layer}-call{call}-{name}.bf16').read_bytes()),dtype=torch.bfloat16).cuda()
from mma_probe import native
def attention(q,k,v,positions):
 return native(q,k,v).reshape(-1,576)
packed=[]
for layer in range(30):
 p=f'model.layers.{layer}.'
 packed.append((p,torch.cat([w[p+'self_attn.'+name+'_proj.weight'] for name in ['q','k','v']],0),
                torch.cat([w[p+'mlp.'+name+'_proj.weight'] for name in ['gate','up']],0)))
keys=[None]*30;vals=[None]*30;tokens=[];ranks=[]
with torch.inference_mode():
 for call in range(32):
  ids=torch.tensor([19556]*128 if call==0 else [tokens[-1]],device='cuda')
  positions=torch.arange(128,device='cuda') if call==0 else torch.tensor([127+call],device='cuda')
  h=w['model.embed_tokens.weight'][ids];res=h
  for layer,(p,qkv,gu) in enumerate(packed):
   q,k,v,res=cp(h,res,w[p+'input_layernorm.weight'],qkv,cos[positions],sin[positions],layer==0)
   keys[layer]=k if keys[layer] is None else torch.cat([keys[layer],k],0)
   vals[layer]=v if vals[layer] is None else torch.cat([vals[layer],v],0)
   context=attention(q,keys[layer],vals[layer],positions)
   h,res=ct(context,res,w[p+'self_attn.o_proj.weight'],w[p+'post_attention_layernorm.weight'],gu,w[p+'mlp.down_proj.weight'])
  logits=torch.nn.functional.linear(norm(h.float()+res.float(),w['model.norm.weight']),w.get('lm_head.weight',w['model.embed_tokens.weight']))
  token=int(logits[-1].argmax().item());tokens.append(token)
  top=logits[-1].float().topk(5);ranks.append({'ids':top.indices.tolist(),'logits':top.values.tolist()})
  print('independent token',call,token,'reference',ref[call],flush=True)
(out/'independent-mma-profile.json').write_text(json.dumps({'tokens':tokens,'reference':ref,'exact':tokens==ref,'top5':ranks,'performance_trials':0},indent=2)+'\n')
