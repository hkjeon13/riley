import ctypes as C,torch,json
from pathlib import Path
from safetensors.torch import load_file
r=Path('/tmp/riley-g04-native-profile-260911');w=load_file('/data/riley-vllm-interim.CfrT9T/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors',device='cuda')
lib=C.CDLL(str(r/'gemv_mma.so'));lib.run.argtypes=[C.c_void_p]*4+[C.c_int]*2
def norm(x,weight):return (x.float()*torch.rsqrt((x.float()*x.float()).mean(-1,keepdim=True)+1e-5)*weight.float()).bfloat16()
norm=torch.compile(norm,fullgraph=True)
x=norm(w['model.embed_tokens.weight'][19556:19557],w['model.layers.0.input_layernorm.weight']);rows=[]
for name in ['q','k','v']:
 weight=w['model.layers.0.self_attn.'+name+'_proj.weight'];out=torch.empty(weight.shape[0],device='cuda',dtype=torch.bfloat16);lib.run(C.c_void_p(torch.cuda.current_stream().cuda_stream),x.data_ptr(),weight.data_ptr(),out.data_ptr(),weight.shape[0],weight.shape[1]);a=torch.nn.functional.linear(x.expand(128,-1).contiguous(),weight)[0];b=torch.nn.functional.linear(x,weight)[0];torch.cuda.synchronize();row={'point':name,'mma_vs_m128':int((out!=a).sum()),'m1_vs_m128':int((a!=b).sum())};rows.append(row);print(row,flush=True)
(r/'gemv-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
