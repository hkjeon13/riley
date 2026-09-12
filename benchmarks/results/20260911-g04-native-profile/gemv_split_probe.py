import ctypes as C,torch,json
from pathlib import Path
from safetensors.torch import load_file
r=Path('/tmp/riley-g04-native-profile-260911');w=load_file('/data/riley-vllm-interim.CfrT9T/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors',device='cuda')
lib=C.CDLL(str(r/'gemv_split.so'));lib.run.argtypes=[C.c_void_p]*4+[C.c_int]*3
def norm(x,weight):return (x.float()*torch.rsqrt((x.float()*x.float()).mean(-1,keepdim=True)+1e-5)*weight.float()).bfloat16()
norm=torch.compile(norm,fullgraph=True)
x=norm(w['model.embed_tokens.weight'][19556:19557],w['model.layers.0.input_layernorm.weight']);rows=[]

weight=torch.cat([w['model.layers.0.self_attn.'+n+'_proj.weight'] for n in ['q','k','v']],0)
a=torch.nn.functional.linear(x.expand(128,-1).contiguous(),weight)[0]
for interval in [0,32,64,128,192,256,288,320,384,512]:
 out=torch.empty(960,device='cuda',dtype=torch.bfloat16)
 lib.run(C.c_void_p(torch.cuda.current_stream().cuda_stream),x.data_ptr(),weight.data_ptr(),out.data_ptr(),960,576,interval)
 torch.cuda.synchronize();row={'interval':interval,'unequal':int((a!=out).sum())};rows.append(row);print(row,flush=True)
(r/'gemv-split-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
