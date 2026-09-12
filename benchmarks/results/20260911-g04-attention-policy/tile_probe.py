import torch,json
from pathlib import Path
r=Path('/tmp/riley-g04-serving-trace-260911');out=Path('/tmp/riley-g04-attention-policy-260911')
torch.backends.cuda.matmul.allow_tf32=False
modes={f'{tile}_{direction}':0 for tile in [64,128,256] for direction in ['forward','reverse']}
records=[]
def read(layer,call,name,heads):
 return torch.frombuffer(bytearray((r/f'layer{layer}-call{call}-{name}.bf16').read_bytes()),dtype=torch.bfloat16).reshape(-1,heads,64).cuda().transpose(0,1)
with torch.inference_mode():
 for layer in range(30):
  K=read(layer,0,'k',3);V=read(layer,0,'v',3)
  for call in range(1,9):
   K=torch.cat([K,read(layer,call,'k',3)],dim=1);V=torch.cat([V,read(layer,call,'v',3)],dim=1)
   q=read(layer,call,'q',9);v=V.repeat_interleave(3,0).float()
   scores=(q.float()@K.repeat_interleave(3,0).float().transpose(-1,-2))*0.125
   target=read(layer,call,'context',9)
   for tile in [64,128,256]:
    for direction in ['forward','reverse']:
     maximum=torch.full((9,1,1),float('-inf'),device='cuda');den=torch.zeros_like(maximum);num=torch.zeros((9,1,64),device='cuda')
     starts=list(range(0,K.shape[1],tile))
     if direction=='reverse':starts.reverse()
     for start in starts:
      s=scores[:,:,start:start+tile];mx=torch.maximum(maximum,s.max(-1,keepdim=True).values)
      alpha=torch.exp2((maximum-mx)*1.4426950408889634)
      p=torch.exp2((s-mx)*1.4426950408889634)
      num=num*alpha+p.bfloat16().float()@v[:,start:start+tile,:]
      den=den*alpha+p.sum(-1,keepdim=True);maximum=mx
     result=(num/den).bfloat16()
     name=f'{tile}_{direction}';neq=int((result!=target).sum().item());modes[name]+=neq
     records.append({'layer':layer,'call':call,'mode':name,'unequal':neq})
(out/'tile-probe.json').write_text(json.dumps({'totals':modes,'records':records,'performance_trials':0},indent=2)+'\n')
print(modes)
