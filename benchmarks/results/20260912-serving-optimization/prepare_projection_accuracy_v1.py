import os,json,pathlib,hashlib
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from transformers import AutoModelForCausalLM
root=pathlib.Path('/tmp/riley-opt-260912/projection-accuracy-v1');root.mkdir(exist_ok=False)
model_path='/data/riley-benchmark/20260827T051948Z-d7ad713a/model'
torch.set_num_threads(8)
m=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True).eval()
xs={};handles=[]
def hook(name):
 def save(mod,args):xs.setdefault(name,[]).append(args[0][0,-1].detach().to(torch.bfloat16).contiguous())
 return save
weights={}
for i,l in enumerate(m.model.layers):
 for name,mod,w in [('qkv',l.self_attn.q_proj,torch.cat([l.self_attn.q_proj.weight,l.self_attn.k_proj.weight,l.self_attn.v_proj.weight])),('gateup',l.mlp.gate_proj,torch.cat([l.mlp.gate_proj.weight,l.mlp.up_proj.weight])),('o',l.self_attn.o_proj,l.self_attn.o_proj.weight),('down',l.mlp.down_proj,l.mlp.down_proj.weight)]:
  key=f'l{i}-{name}';weights[key]=w.detach().to(torch.bfloat16).contiguous();handles.append(mod.register_forward_pre_hook(hook(key)))
weights['head']=m.lm_head.weight.detach().to(torch.bfloat16).contiguous();handles.append(m.lm_head.register_forward_pre_hook(hook('head')))
lengths=[16,32,64,128,192,256,384,512]
with torch.inference_mode():
 for j,n in enumerate(lengths):
  ids=[(i*613+j*1879+29)%49152 for i in range(n)];m(input_ids=torch.tensor([ids]),use_cache=False);print('captured',n,flush=True)
for h in handles:h.remove()
manifest={};lines=[]
for key,w in weights.items():
 x=torch.stack(xs[key]);assert x.shape==(8,w.shape[1])
 for suffix,t in [('w',w),('x',x)]:
  p=root/f'{key}-{suffix}.bin';p.write_bytes(t.view(torch.uint16).numpy().tobytes());manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 lines.append(f'{key}\t{w.shape[0]}\t{w.shape[1]}\t{root/key}-w.bin\t{root/key}-x.bin')
(root/'cases.tsv').write_text('\n'.join(lines)+'\n')
(root/'manifest.json').write_text(json.dumps({'source':'CPU FP32 HuggingFace eager model last-token projection inputs rounded to BF16; synthetic held-out token streams','prompt_lengths':lengths,'batch_rows':8,'cases':len(lines),'model_sha256':hashlib.sha256(pathlib.Path(model_path+'/model.safetensors').read_bytes()).hexdigest(),'files':manifest},indent=2)+'\n')
print('complete',len(lines),flush=True)
