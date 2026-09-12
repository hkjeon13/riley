import ctypes as C,torch,json
from pathlib import Path
from safetensors.torch import load_file
r=Path('/tmp/riley-g04-native-profile-260911');w=load_file('/data/riley-vllm-interim.CfrT9T/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors',device='cuda')
lib=C.CDLL(str(r/'gemv_split.so'));lib.run.argtypes=[C.c_void_p]*4+[C.c_int]*3
def norm(x,weight):return (x.float()*torch.rsqrt((x.float()*x.float()).mean(-1,keepdim=True)+1e-5)*weight.float()).bfloat16()
norm=torch.compile(norm,fullgraph=True)
x=norm(w['model.embed_tokens.weight'][19556:19557],w['model.layers.0.input_layernorm.weight']);rows=[]


torch.manual_seed(19)
for layer in range(30):
 p=f'model.layers.{layer}.'
 specs=[('qkv',torch.cat([w[p+'self_attn.'+n+'_proj.weight'] for n in ['q','k','v']],0),192),('o',w[p+'self_attn.o_proj.weight'],128),('gate_up',torch.cat([w[p+'mlp.'+n+'_proj.weight'] for n in ['gate','up']],0),0),('down',w[p+'mlp.down_proj.weight'],320)]
 for name,weight,interval in specs:
  inputs=torch.randn(128,weight.shape[1],device='cuda',dtype=torch.bfloat16);out=torch.empty(weight.shape[0],device='cuda',dtype=torch.bfloat16)
  ref=torch.nn.functional.linear(inputs,weight)[0]
  lib.run(C.c_void_p(torch.cuda.current_stream().cuda_stream),inputs.data_ptr(),weight.data_ptr(),out.data_ptr(),weight.shape[0],weight.shape[1],interval)
  torch.cuda.synchronize();row={'layer':layer,'point':name,'interval':interval,'unequal':int((out!=ref).sum())};rows.append(row)
  if row['unequal']:print(row,flush=True)
print('cases',len(rows),'unequal',sum(x['unequal'] for x in rows),flush=True)
(r/'gemv-all-shapes.json').write_text(json.dumps(rows,indent=2)+'\n')
