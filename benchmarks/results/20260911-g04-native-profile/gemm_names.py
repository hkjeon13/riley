import ctypes as C,torch,json
from pathlib import Path
from safetensors.torch import load_file
r=Path('/tmp/riley-g04-native-profile-260911');w=load_file('/data/riley-vllm-interim.CfrT9T/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors',device='cuda')
lib=C.CDLL(str(r/'gemv_round.so'));lib.run.argtypes=[C.c_void_p]*4+[C.c_int]*3
def norm(x,weight):return (x.float()*torch.rsqrt((x.float()*x.float()).mean(-1,keepdim=True)+1e-5)*weight.float()).bfloat16()
norm=torch.compile(norm,fullgraph=True)
x=norm(w['model.embed_tokens.weight'][19556:19557],w['model.layers.0.input_layernorm.weight']);rows=[]

weight=torch.cat([w['model.layers.0.self_attn.'+n+'_proj.weight'] for n in ['q','k','v']],0)

input128=x.expand(128,-1).contiguous()
for _ in range(3): torch.nn.functional.linear(input128,weight)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 y=torch.nn.functional.linear(input128,weight)
torch.cuda.synchronize()
names=sorted(set(e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA))
print(json.dumps(names,indent=2),flush=True)
(r/'prefill-gemm-kernel-names.json').write_text(json.dumps({'kernel_names':names,'purpose':'operator identity only; no timing results retained','performance_trials':0},indent=2)+'\n')

trace=r/'temporary-operator-trace.json';prof.export_chrome_trace(str(trace))
events=json.loads(trace.read_text())['traceEvents']
grids=[{'name':e['name'],'args':e.get('args',{})} for e in events if e.get('cat')=='kernel']
print(json.dumps(grids,indent=2),flush=True)
(r/'prefill-gemm-launch-shapes.json').write_text(json.dumps(grids,indent=2)+'\n');trace.unlink()
